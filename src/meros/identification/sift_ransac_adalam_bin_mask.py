from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.identification.common import pairs, save_matches
from meros.identification.sift_ransac_adalam import (
    ComparisonResult,
    Features,
    compare,
    prepare_photo,
)
from meros.project import Project
from meros.utils import configure_torch

OUTPUT = Path("data/media/sandbox/sift_ransac_adalam_bin_mask/results.csv")

MIN_INLIERS = 20

SHOW_PLOTS = False


@dataclass
class AlignmentResult:
    matrix: np.ndarray
    inliers: np.ndarray
    matches: list[cv.DMatch]
    kp_a: list[cv.KeyPoint]
    kp_b: list[cv.KeyPoint]
    errors: np.ndarray

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    @property
    def num_inliers(self) -> int:
        return int(np.count_nonzero(self.inliers))

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / max(self.num_matches, 1)

    @property
    def median_error(self) -> float:
        return float(np.median(self.errors[self.inliers])) if self.num_inliers else float("nan")


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    return image


def body_silhouette(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        return (image[:, :, 3] > 0).astype(np.uint8)

    bgr = to_bgr(image)
    return np.any(bgr > 3, axis=2).astype(np.uint8)


def load_pattern(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape

    mask = cv.resize(
        mask,
        (width, height),
        interpolation=cv.INTER_NEAREST,
    )

    return mask > 127


def is_valid_homography(
    H: np.ndarray,
    shape: tuple[int, int],
    *,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 10.0,
) -> bool:
    height, width = shape

    corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    transformed = cv.perspectiveTransform(corners, H)  # pyright: ignore

    if not np.all(np.isfinite(transformed)):
        return False

    points = transformed.reshape(-1, 2)

    area = abs(cv.contourArea(points))
    original_area = width * height

    ratio = area / original_area

    if not min_area_ratio <= ratio <= max_area_ratio:
        return False

    # Check that the projected quadrilateral is convex.
    contour = points.reshape(-1, 1, 2).astype(np.float32)

    if not cv.isContourConvex(contour):
        return False

    return True


def inlier_coverage(
    keypoints: list[cv.KeyPoint],
    matches: list[cv.DMatch],
    inliers: np.ndarray,
    overlap: np.ndarray,
) -> float:
    points = np.array(
        [keypoints[m.trainIdx].pt for m, valid in zip(matches, inliers, strict=True) if valid],  # pyright: ignore[reportAttributeAccessIssue]
        dtype=np.float32,
    )

    if len(points) < 3:
        return 0.0

    hull = cv.convexHull(points)

    hull_mask = np.zeros(
        overlap.shape,
        dtype=np.uint8,
    )

    cv.fillConvexPoly(
        hull_mask,  # pyright: ignore
        np.round(hull.reshape(-1, 2)).astype(np.int32),
        1,
    )

    overlap = overlap.astype(bool)

    hull_area = np.count_nonzero(hull_mask.astype(bool) & overlap)

    overlap_area = np.count_nonzero(overlap)

    if overlap_area == 0:
        return 0.0

    return hull_area / overlap_area


def estimate_alignment(
    first: Features,
    second: Features,
    comparison: ComparisonResult,
) -> AlignmentResult | None:
    matrix = comparison.homography

    if matrix is None:
        return None

    matches = comparison.matches

    src = np.asarray(
        [first.keypoints[m.queryIdx].pt for m in matches],  # pyright: ignore[reportAttributeAccessIssue]
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    dst = np.asarray(
        [second.keypoints[m.trainIdx].pt for m in matches],  # pyright: ignore[reportAttributeAccessIssue]
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    projected = cv.perspectiveTransform(src, matrix)  # pyright: ignore

    errors = np.linalg.norm(
        projected.reshape(-1, 2) - dst.reshape(-1, 2),
        axis=1,
    )

    return AlignmentResult(
        matrix=matrix,
        inliers=comparison.global_inliers,
        matches=matches,
        kp_a=second.keypoints,
        kp_b=first.keypoints,
        errors=errors,
    )


def warp(
    image: np.ndarray,
    matrix: np.ndarray,
    shape: tuple[int, int],
    *,
    nearest: bool = False,
) -> np.ndarray:
    height, width = shape

    interpolation = cv.INTER_NEAREST if nearest else cv.INTER_LINEAR

    if matrix.shape == (3, 3):
        return cv.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=interpolation,
        )

    return cv.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
    )


def visualize_matches(
    image_a: np.ndarray,
    image_b: np.ndarray,
    result: AlignmentResult,
) -> None:
    inlier_matches = [
        match
        for match, valid in zip(
            result.matches,
            result.inliers,
            strict=True,
        )
        if valid
    ]

    visualization = cv.drawMatches(
        to_bgr(image_b),
        result.kp_b,
        to_bgr(image_a),
        result.kp_a,
        inlier_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(18, 8))

    plt.imshow(cv.cvtColor(visualization, cv.COLOR_BGR2RGB))

    plt.title(
        f"RANSAC inliers: {result.num_inliers} / {result.num_matches} ({result.inlier_ratio:.1%})"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_alignment(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    silhouette_a: np.ndarray,
    silhouette_b: np.ndarray,
    result: AlignmentResult,
) -> tuple[float, float, float, float]:
    height, width = image_a.shape[:2]
    shape = (height, width)

    aligned_b = warp(image_b, result.matrix, shape)

    aligned_pattern_b = warp(
        pattern_b.astype(np.uint8),
        result.matrix,
        shape,
        nearest=True,
    ).astype(bool)

    aligned_silhouette_b = warp(
        silhouette_b,
        result.matrix,
        shape,
        nearest=True,
    ).astype(bool)

    # Valid pixels from B after geometric transformation.
    valid_b = warp(
        np.ones(image_b.shape[:2], dtype=np.uint8),
        result.matrix,
        shape,
        nearest=True,
    ).astype(bool)

    # Only compare pixels belonging to both animal silhouettes.
    overlap = silhouette_a.astype(bool) & aligned_silhouette_b & valid_b

    rgb_a = cv.cvtColor(to_bgr(image_a), cv.COLOR_BGR2RGB)
    rgb_b = cv.cvtColor(to_bgr(aligned_b), cv.COLOR_BGR2RGB)

    # RGB overlay only where B is valid.
    overlay = rgb_a.copy()

    blended = cv.addWeighted(rgb_a, 0.5, rgb_b, 0.5, 0)

    overlay[overlap] = blended[overlap]

    # Pattern comparison restricted to the shared animal region.
    binary_a = pattern_a & overlap
    binary_b = aligned_pattern_b & overlap

    intersection = binary_a & binary_b
    union = binary_a | binary_b

    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)

    iou = intersection_count / union_count if union_count > 0 else float("nan")

    coverage_a = np.count_nonzero(overlap) / max(np.count_nonzero(silhouette_a), 1)

    coverage_b = np.count_nonzero(overlap) / max(np.count_nonzero(aligned_silhouette_b), 1)

    # Red = A only, green = B only, yellow = agreement.
    pattern_overlay = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    inlier_cov = inlier_coverage(result.kp_a, result.matches, result.inliers, overlap)

    pattern_overlay[binary_a, 0] = 255
    pattern_overlay[binary_b, 1] = 255

    # Show the shared region on the reference image.
    overlap_visualization = rgb_a.copy()
    overlap_visualization[~overlap] = (overlap_visualization[~overlap] * 0.2).astype(np.uint8)

    if SHOW_PLOTS:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(rgb_a)
        axes[0, 0].set_title("Image A (reference)")

        axes[0, 1].imshow(rgb_b)
        axes[0, 1].set_title("Image B (aligned)")

        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title("RGB overlay (shared region)")

        axes[1, 0].imshow(overlap_visualization)
        axes[1, 0].set_title("Shared animal region")

        axes[1, 1].imshow(pattern_overlay)
        axes[1, 1].set_title("Patterns: red=A, green=B, yellow=both")

        axes[1, 2].imshow(pattern_overlay)
        axes[1, 2].imshow(
            overlap.astype(np.uint8),
            cmap="gray",
            alpha=0.18,
            vmin=0,
            vmax=1,
        )
        axes[1, 2].set_title(f"Pattern IoU: {iou:.3f}")

        for ax in axes.flat:
            ax.axis("off")

        fig.suptitle(
            f"Matches: {result.num_matches} | "
            f"Inliers: {result.num_inliers} | "
            f"Ratio: {result.inlier_ratio:.1%} | "
            f"Median error: {result.median_error:.2f}px",
            fontsize=14,
        )

        plt.tight_layout()
        plt.show()

    return coverage_a, coverage_b, iou, inlier_cov


def run(project: Project) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)

    configure_torch()

    for subject_a, subject_b, a, b in pairs(
        project, a_filter="rogerio", b_filter="rogerio", bin_filter=False
    ):
        image_a = a.crop_im
        image_b = b.crop_im

        mask_a = a.binary_im
        mask_b = b.binary_im

        assert image_a is not None
        assert image_b is not None
        assert mask_a is not None
        assert mask_b is not None

        first = prepare_photo(f"{subject_a._id}/{a._id}", image_a)
        second = prepare_photo(f"{subject_b._id}/{b._id}", image_b)

        comparison = compare(first, second)
        result = estimate_alignment(first, second, comparison)

        matches = len(comparison.matches)
        global_inliers = int(comparison.global_inliers.sum())
        local_inliers = int(comparison.local_inliers.sum())
        both_inliers = int((comparison.global_inliers & comparison.local_inliers).sum())

        if result is None:
            status = "no_homography"
        elif result.num_inliers < MIN_INLIERS:
            status = "insufficient_inliers"
        elif not is_valid_homography(result.matrix, first.gray.shape):
            status = "invalid_homography"
        else:
            status = "ok"

        metrics: dict[str, float] = {}

        if status == "ok":
            # The legacy homography maps image A to image B.
            # Evaluate the patterns in image B's reference coordinate system.
            resized_a = cv.resize(
                image_a,
                (first.gray.shape[1], first.gray.shape[0]),
                interpolation=cv.INTER_AREA,
            )
            resized_b = cv.resize(
                image_b,
                (second.gray.shape[1], second.gray.shape[0]),
                interpolation=cv.INTER_AREA,
            )

            pattern_a = load_pattern(mask_a, first.gray.shape)
            pattern_b = load_pattern(mask_b, second.gray.shape)

            silhouette_a = cv.resize(
                body_silhouette(image_a),
                (first.gray.shape[1], first.gray.shape[0]),
                interpolation=cv.INTER_NEAREST,
            )
            silhouette_b = cv.resize(
                body_silhouette(image_b),
                (second.gray.shape[1], second.gray.shape[0]),
                interpolation=cv.INTER_NEAREST,
            )

            assert result is not None

            if SHOW_PLOTS:
                visualize_matches(resized_b, resized_a, result)

            coverage_b, coverage_a, iou, inlier_cov = visualize_alignment(
                resized_b,
                resized_a,
                pattern_b,
                pattern_a,
                silhouette_b,
                silhouette_a,
                result,
            )

            metrics = {
                "coverage_a": coverage_a,
                "coverage_b": coverage_b,
                "pattern_iou": iou,
                "inlier_coverage": inlier_cov,
            }

        save_matches(
            OUTPUT,
            method="sift_ransac_adalam_bin_mask",
            subject_a=subject_a._id,
            image_a=a._id,
            subject_b=subject_b._id,
            image_b=b._id,
            status=status,
            matches=matches,
            global_inliers=global_inliers,
            local_inliers=local_inliers,
            both_inliers=both_inliers,
            median_error=result.median_error if result is not None and result.num_inliers else None,
            **metrics,
        )

        print(
            f"{first.name} vs {second.name}: {status}, "
            f"global {global_inliers}, local {local_inliers} / {matches}",
            flush=True,
        )


if __name__ == "__main__":
    run(Project.open())
