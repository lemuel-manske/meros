"""Measure whether spot appearance improves candidate retrieval on known pairs."""

from __future__ import annotations

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree

from meros.processing.compare_spot_geometry import neighborhood_descriptors
from meros.project import Project


SUBJECT_ID = "luana"
FIRST_IMAGE_ID = 0
SECOND_IMAGE_ID = 1
PAIR_RADIUS_PX = 15.0
PATCH_EYE_FRACTION = 0.4
PATCH_SIZE = 21
APPEARANCE_WEIGHT = 0.5
RETRIEVAL_K = (1, 5, 10)


def appearance_descriptors(
    image: np.ndarray, points: np.ndarray, eye_radius: float
) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    radius = max(5, round(eye_radius * PATCH_EYE_FRACTION))
    gray = cv.copyMakeBorder(
        gray, radius, radius, radius, radius, cv.BORDER_REFLECT_101
    )

    features = []

    for x, y in points:
        cx, cy = round(float(x)) + radius, round(float(y)) + radius
        patch = gray[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
        patch = cv.resize(patch, (PATCH_SIZE, PATCH_SIZE)).astype(np.float32)

        # Local contrast normalization removes overall exposure differences.
        patch = (patch - patch.mean()) / max(float(patch.std()), 10.0)

        # A radial profile is robust to rotations and retains the dark center
        # and its surrounding skin. It does not require a spot mask.
        yy, xx = np.indices(patch.shape)
        distance = np.hypot(xx - PATCH_SIZE // 2, yy - PATCH_SIZE // 2)
        bands = np.linspace(0, PATCH_SIZE / 2, 7)
        feature = [
            float(patch[(distance >= lo) & (distance < hi)].mean())
            for lo, hi in zip(bands[:-1], bands[1:], strict=True)
        ]
        features.append(feature)

    return np.asarray(features, dtype=np.float32)


def ranks(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return np.argsort(distances, axis=1)


def run(project: Project) -> None:
    image_a = project.media.read_subject_image(SUBJECT_ID, FIRST_IMAGE_ID)
    image_b = project.media.read_subject_image(SUBJECT_ID, SECOND_IMAGE_ID)
    points_a = np.asarray(
        project.metadata.read_subject_spots(SUBJECT_ID, FIRST_IMAGE_ID),
        dtype=np.float32,
    )
    points_b = np.asarray(
        project.metadata.read_subject_spots(SUBJECT_ID, SECOND_IMAGE_ID),
        dtype=np.float32,
    )
    eye_a = project.metadata.read_subject_eye(SUBJECT_ID, FIRST_IMAGE_ID)
    eye_b = project.metadata.read_subject_eye(SUBJECT_ID, SECOND_IMAGE_ID)

    radius_a = float(np.sqrt(eye_a.radius_x * eye_a.radius_y))
    radius_b = float(np.sqrt(eye_b.radius_x * eye_b.radius_y))

    geometry_a = neighborhood_descriptors(points_a)
    geometry_b = neighborhood_descriptors(points_b)
    appearance_a = appearance_descriptors(image_a, points_a, radius_a)
    appearance_b = appearance_descriptors(image_b, points_b, radius_b)

    # Normalize descriptor distances so the mixing weight is interpretable.
    geometry_dist = np.linalg.norm(
        geometry_a[:, None] - geometry_b[None, :], axis=2
    )
    appearance_dist = np.linalg.norm(
        appearance_a[:, None] - appearance_b[None, :], axis=2
    )
    geometry_dist /= max(float(np.median(geometry_dist)), 1e-6)
    appearance_dist /= max(float(np.median(appearance_dist)), 1e-6)

    rankings = {
        "Geometry": np.argsort(geometry_dist, axis=1),
        "Appearance": np.argsort(appearance_dist, axis=1),
        "Combined": np.argsort(
            (1 - APPEARANCE_WEIGHT) * geometry_dist
            + APPEARANCE_WEIGHT * appearance_dist,
            axis=1,
        ),
    }

    pairs = project.metadata.read_subject_spot_correspondences(
        SUBJECT_ID, FIRST_IMAGE_ID, SECOND_IMAGE_ID
    )
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)
    known = []

    for pair in pairs:
        da, ia = tree_a.query(pair["first"])
        db, ib = tree_b.query(pair["second"])

        if max(da, db) <= PAIR_RADIUS_PX:
            known.append((int(ia), int(ib)))

    print(f"Known pairs represented by annotations: {len(known)}/{len(pairs)}")

    for label, ranking in rankings.items():
        print(f"\\n{label}")

        for k in RETRIEVAL_K:
            hits = sum(j in ranking[i, :k] for i, j in known)
            print(f"  recall@{k}: {hits}/{len(known)}")

    print(
        "\\nDiagnostic only: these are candidates for the Luana positive pair; "
        "no claims of improved identification or geometric consensus."
    )


if __name__ == "__main__":
    run(Project.open())
