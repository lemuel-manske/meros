"""Diagnose why known spot correspondences are missed by geometric matching."""

from __future__ import annotations

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree

from meros.project import Project


SUBJECT_ID = "luana"
FIRST_IMAGE_ID = 0
SECOND_IMAGE_ID = 1

ANNOTATION_RADIUS_PX = 12.0
RANSAC_THRESHOLD_PX = 12.0
DESCRIPTOR_NEIGHBORS = 6
RETRIEVAL_K = (1, 5, 10)


def descriptors(points: np.ndarray) -> np.ndarray:
    """Ordered neighbor distances, normalized by each point's local scale."""
    distances = cKDTree(points).query(points, k=DESCRIPTOR_NEIGHBORS + 1)[0][:, 1:]
    scale = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-6)

    return distances / scale


def fit_transform(
    name: str, source: np.ndarray, target: np.ndarray, affine: bool
) -> None:
    estimator = cv.estimateAffine2D if affine else cv.estimateAffinePartial2D

    matrix, inliers = estimator(
        source.astype(np.float32),
        target.astype(np.float32),
        method=cv.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=2000,
        confidence=0.99,
    )

    if matrix is None:
        print(f"  {name}: no transform")
        return

    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    errors = np.linalg.norm(predicted - target, axis=1)

    print(
        f"  {name}: inliers={int(inliers.sum())}/{len(source)}, "
        f"median_error={np.median(errors):.1f}px, "
        f"median_inlier_error={np.median(errors[inliers.ravel() > 0]):.1f}px"
    )
    print("    residuals (annotation order):", np.round(errors, 1).tolist())


def run(project: Project) -> None:
    pairs = project.metadata.read_subject_spot_correspondences(
        SUBJECT_ID, FIRST_IMAGE_ID, SECOND_IMAGE_ID
    )
    first = np.asarray([pair["first"] for pair in pairs], dtype=np.float64)
    second = np.asarray([pair["second"] for pair in pairs], dtype=np.float64)

    first_spots = np.asarray(
        project.metadata.read_subject_spots(SUBJECT_ID, FIRST_IMAGE_ID),
        dtype=np.float64,
    )
    second_spots = np.asarray(
        project.metadata.read_subject_spots(SUBJECT_ID, SECOND_IMAGE_ID),
        dtype=np.float64,
    )

    print(
        f"{SUBJECT_ID}/{FIRST_IMAGE_ID}-{SECOND_IMAGE_ID}: "
        f"{len(pairs)} known pairs, "
        f"{len(first_spots)} / {len(second_spots)} independently annotated spots"
    )

    distance_a, index_a = cKDTree(first_spots).query(first)
    distance_b, index_b = cKDTree(second_spots).query(second)

    available = (distance_a <= ANNOTATION_RADIUS_PX) & (
        distance_b <= ANNOTATION_RADIUS_PX
    )

    print(
        f"Known pairs represented by existing spot annotations "
        f"(<= {ANNOTATION_RADIUS_PX:g}px on each side): "
        f"{int(available.sum())}/{len(pairs)}"
    )

    print("  Missing pair IDs:", (np.flatnonzero(~available) + 1).tolist())

    print("\nGeometry of manually established correspondences:")
    fit_transform("Similarity", first, second, affine=False)
    fit_transform("Affine", first, second, affine=True)

    descriptor_a = descriptors(first_spots)
    descriptor_b = descriptors(second_spots)
    ranked = cKDTree(descriptor_b).query(
        descriptor_a[index_a[available]],
        k=min(max(RETRIEVAL_K), len(second_spots)),
    )[1]
    ranked = np.atleast_2d(ranked)

    ground_truth = index_b[available]

    print("\nCorrect-match retrieval by local distance descriptor:")
    print(f"  Eligible known pairs: {len(ground_truth)}")

    for k in RETRIEVAL_K:
        hits = np.any(ranked[:, :k] == ground_truth[:, None], axis=1)
        print(f"  recall@{k}: {int(hits.sum())}/{len(ground_truth)}")

    print(
        "\nThese are diagnostics on one known-positive pair, "
        "not identification accuracy on independent subjects."
    )


if __name__ == "__main__":
    run(Project.open())
