"""Compare geometric transforms fitted to manually verified spot correspondences.

This measures how well annotated points in one photo predict their locations in
another. It does not test automatic spot matching or identity recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from meros.project import Project


RANSAC_THRESHOLD_PX = 12.0
SAVE_OVERLAYS = False
OUTPUT_DIR = Path("results/spot_geometry")


@dataclass(frozen=True)
class TransformModel:
    name: str
    minimum_points: int


MODELS = (
    TransformModel("similarity", 2),
    TransformModel("affine", 3),
    TransformModel("homography", 4),
)


def estimate(
    source: np.ndarray,
    target: np.ndarray,
    model: TransformModel,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(source) < model.minimum_points:
        return None

    try:
        return _estimate_opencv(source, target, model)
    except cv.error:
        return None


def _estimate_opencv(
    source: np.ndarray,
    target: np.ndarray,
    model: TransformModel,
) -> tuple[np.ndarray, np.ndarray] | None:
    if model.name == "similarity":
        matrix, mask = cv.estimateAffinePartial2D(
            source, target, method=cv.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PX
        )
    elif model.name == "affine":
        matrix, mask = cv.estimateAffine2D(
            source, target, method=cv.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PX
        )
    else:
        matrix, mask = cv.findHomography(source, target, cv.RANSAC, RANSAC_THRESHOLD_PX)

    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None

    if model.name != "homography":
        matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))

    return matrix, mask.ravel().astype(bool)


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = cv.perspectiveTransform(
            np.asarray(points, dtype=np.float64).reshape(1, -1, 2), matrix
        ).reshape(-1, 2)

    return projected


def leave_one_out_errors(
    source: np.ndarray,
    target: np.ndarray,
    model: TransformModel,
) -> np.ndarray:
    # A held-out spot must never contribute to its own transform.
    errors = np.full(len(source), np.nan)

    if len(source) <= model.minimum_points:
        return errors

    for held_out in range(len(source)):
        retained = np.arange(len(source)) != held_out
        fit = estimate(source[retained], target[retained], model)

        if fit is None:
            continue

        projected = project_points(source[held_out:held_out + 1], fit[0])

        if np.isfinite(projected).all():
            errors[held_out] = np.linalg.norm(projected[0] - target[held_out])

    return errors


def draw_overlay(
    project: Project,
    subject_id: str,
    first_id: int,
    second_id: int,
    matrix: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    name: str,
) -> None:
    first = project.media.read_subject_image(subject_id, first_id)
    second = project.media.read_subject_image(subject_id, second_id)
    height, width = second.shape[:2]

    warped = cv.warpPerspective(first, matrix, (width, height))
    coverage = cv.warpPerspective(
        np.full(first.shape[:2], 255, dtype=np.uint8), matrix, (width, height)
    ) > 0

    image = second.copy()
    image[coverage] = cv.addWeighted(second[coverage], 0.5, warped[coverage], 0.5, 0)

    projected = project_points(source, matrix)

    for expected, observed in zip(projected, target, strict=True):
        if not np.isfinite(expected).all():
            continue

        expected_point = tuple(np.rint(expected).astype(int))
        observed_point = tuple(np.rint(observed).astype(int))

        cv.line(image, expected_point, observed_point, (0, 255, 255), 1)
        cv.circle(image, expected_point, 6, (0, 255, 0), 2)
        cv.circle(image, observed_point, 4, (0, 0, 255), -1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{subject_id}_{first_id}-{second_id}_{name}.png"
    cv.imwrite(str(destination), image)
    print(f"    Overlay: {destination}")


def evaluate_pair(
    project: Project,
    subject_id: str,
    first_id: int,
    second_id: int,
) -> None:
    annotations = project.metadata.read_subject_spot_correspondences(
        subject_id, first_id, second_id
    )

    if not annotations:
        return

    source = np.asarray([pair["first"] for pair in annotations], dtype=np.float64)
    target = np.asarray([pair["second"] for pair in annotations], dtype=np.float64)

    print(f"\n{subject_id}/{first_id}-{second_id}: {len(source)} annotated pairs")

    candidates = []

    for model in MODELS:
        fit = estimate(source, target, model)

        if fit is None:
            print(f"  {model.name}: insufficient points or degenerate fit")
            continue

        matrix, inliers = fit
        projected = project_points(source, matrix)
        training_errors = np.linalg.norm(projected - target, axis=1)
        held_out_errors = leave_one_out_errors(source, target, model)
        valid = held_out_errors[np.isfinite(held_out_errors)]

        if not np.isfinite(training_errors).all():
            print(f"  {model.name}: unstable projection")
            continue

        if len(valid):
            median = float(np.median(valid))
            p90 = float(np.percentile(valid, 90))
            cross_validation = (
                f"LOO median={median:.1f}px p90={p90:.1f}px "
                f"({len(valid)}/{len(source)})"
            )
            candidates.append((median, p90, model.name, matrix))
        else:
            cross_validation = "LOO unavailable"

        print(
            f"  {model.name}: inliers={int(inliers.sum())}/{len(source)}, "
            f"fit median={np.median(training_errors):.1f}px, {cross_validation}"
        )

    if not candidates:
        return

    _, _, best_name, best_matrix = min(candidates)
    print(f"  Best held-out median: {best_name}")

    if SAVE_OVERLAYS:
        draw_overlay(
            project,
            subject_id,
            first_id,
            second_id,
            best_matrix,
            source,
            target,
            best_name,
        )


def run(project: Project) -> None:
    # Discover all subjects, including those without locally downloaded photos.
    root = Path("data/metadata/subjects")
    paths = sorted(root.glob("*/spot_correspondences/*.json"))

    if not paths:
        print(f"No spot correspondences found under {root}")
        return

    for path in paths:
        first_id, second_id = (int(value) for value in path.stem.split("-"))
        subject_id = path.parent.parent.name
        evaluate_pair(project, subject_id, first_id, second_id)


if __name__ == "__main__":
    run(Project.open())
