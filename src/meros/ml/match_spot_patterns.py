"""
Propose spot correspondences using manually annotated eye and mouth landmarks.

The matcher only sees point coordinates and anatomical anchors, never subject IDs
or manually verified spot-to-spot correspondences. This is an exploratory baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from meros.project import Project


SOURCE = ("luana", 0)
TARGET = ("luana", 1)
MATCH_RADIUS_EYE_MOUTH = 0.12
SHOW_PLOT = True


@dataclass(frozen=True)
class SpotMatch:
    matrix: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    errors: tuple[float, ...]


def initial_similarity(
    source_eye: tuple[float, float],
    source_mouth: tuple[float, float],
    target_eye: tuple[float, float],
    target_mouth: tuple[float, float],
) -> np.ndarray:
    source_origin = np.asarray(source_eye, dtype=float)
    target_origin = np.asarray(target_eye, dtype=float)
    source_axis = np.asarray(source_mouth, dtype=float) - source_origin
    target_axis = np.asarray(target_mouth, dtype=float) - target_origin

    source_length = np.linalg.norm(source_axis)
    target_length = np.linalg.norm(target_axis)

    if source_length == 0 or target_length == 0:
        raise ValueError("Eye and mouth landmarks must be distinct")

    cosine = np.dot(source_axis, target_axis) / (source_length * target_length)
    sine = np.linalg.det(np.stack((source_axis, target_axis))) / (
        source_length * target_length
    )
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    linear = rotation * (target_length / source_length)

    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = target_origin - linear @ source_origin

    return matrix


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(1, -1, 2), matrix
    ).reshape(-1, 2)


def assign_spots(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    radius: float,
) -> tuple[tuple[int, int], ...]:
    predicted = project_points(source, matrix)
    distances = cdist(predicted, target)
    source_indices, target_indices = np.where(distances <= radius)
    order = np.argsort(distances[source_indices, target_indices], kind="stable")

    used_source: set[int] = set()
    used_target: set[int] = set()
    pairs = []

    for position in order:
        source_id = int(source_indices[position])
        target_id = int(target_indices[position])

        if source_id in used_source or target_id in used_target:
            continue

        pairs.append((source_id, target_id))
        used_source.add(source_id)
        used_target.add(target_id)

    return tuple(pairs)


def match_spots(
    source: np.ndarray,
    target: np.ndarray,
    source_eye: tuple[float, float],
    source_mouth: tuple[float, float],
    target_eye: tuple[float, float],
    target_mouth: tuple[float, float],
) -> SpotMatch:
    matrix = initial_similarity(source_eye, source_mouth, target_eye, target_mouth)
    eye_mouth_distance = np.linalg.norm(np.asarray(target_mouth) - np.asarray(target_eye))
    radius = float(eye_mouth_distance * MATCH_RADIUS_EYE_MOUTH)

    pairs = assign_spots(source, target, matrix, radius)

    if len(pairs) >= 3:
        source_ids, target_ids = zip(*pairs, strict=True)
        try:
            refined, inliers = cv.estimateAffine2D(
                source[list(source_ids)],
                target[list(target_ids)],
                method=cv.RANSAC,
                ransacReprojThreshold=radius,
            )
        except cv.error:
            refined, inliers = None, None

        if refined is not None and inliers is not None and int(inliers.sum()) >= 3:
            candidate = np.vstack((refined, [0.0, 0.0, 1.0]))
            candidate_pairs = assign_spots(source, target, candidate, radius)

            if len(candidate_pairs) >= len(pairs):
                matrix = candidate
                pairs = candidate_pairs

    predicted = project_points(source, matrix)
    errors = tuple(float(np.linalg.norm(predicted[i] - target[j])) for i, j in pairs)

    return SpotMatch(matrix=matrix, pairs=pairs, errors=errors)


def show_match(
    project: Project,
    source_label: tuple[str, int],
    target_label: tuple[str, int],
    source: np.ndarray,
    target: np.ndarray,
    match: SpotMatch,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    predicted = project_points(source, match.matrix)

    for axis, (subject_id, image_id), points in zip(
        axes, (source_label, target_label), (source, target), strict=True
    ):
        image = project.media.read_subject_image(subject_id, image_id)
        axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axis.scatter(points[:, 0], points[:, 1], s=16, c="yellow")
        axis.set_title(f"{subject_id}/{image_id}")
        axis.set_axis_off()

    for first, second in match.pairs:
        axes[0].scatter(source[first, 0], source[first, 1], s=36, c="cyan")
        axes[1].plot(
            [predicted[first, 0], target[second, 0]],
            [predicted[first, 1], target[second, 1]],
            c="red",
            linewidth=1,
        )

    if match.pairs:
        source_ids, target_ids = zip(*match.pairs, strict=True)
        axes[1].scatter(
            target[list(target_ids), 0],
            target[list(target_ids), 1],
            s=36,
            c="cyan",
            label="Proposed matches",
        )
        axes[1].scatter(
            predicted[list(source_ids), 0],
            predicted[list(source_ids), 1],
            s=40,
            c="lime",
            marker="x",
            label="Projected",
        )
        axes[1].legend(loc="upper right")

    figure.suptitle(f"{len(match.pairs)} proposed spot pairs (not verified)")
    figure.tight_layout()
    plt.show()


def run(project: Project) -> None:
    source_eye = project.metadata.read_subject_eye(*SOURCE)
    target_eye = project.metadata.read_subject_eye(*TARGET)
    source_mouth = project.metadata.read_subject_mouth(*SOURCE)
    target_mouth = project.metadata.read_subject_mouth(*TARGET)

    if (
        source_eye is None
        or target_eye is None
        or source_mouth is None
        or target_mouth is None
    ):
        print("Both images need manually annotated eye and mouth landmarks.")
        return

    source = np.asarray(project.metadata.read_subject_spots(*SOURCE), dtype=np.float64)
    target = np.asarray(project.metadata.read_subject_spots(*TARGET), dtype=np.float64)

    if len(source) == 0 or len(target) == 0:
        print("Both images need manually annotated spot centers.")
        return

    result = match_spots(
        source,
        target,
        source_eye.center,
        source_mouth,
        target_eye.center,
        target_mouth,
    )

    print(f"{SOURCE} vs {TARGET}: {len(result.pairs)} proposed correspondences")

    if result.errors:
        print(f"Median reprojection error: {np.median(result.errors):.1f}px")

    if SHOW_PLOT:
        show_match(project, SOURCE, TARGET, source, target, result)


if __name__ == "__main__":
    run(Project.open())
