"""Diagnose where joint matching loses the known Luana correspondences."""

from __future__ import annotations

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree

from meros.processing.compare_joint_spot_matching import (
    CANDIDATES_PER_SPOT,
    CONSENSUS_RADIUS,
    KNOWN_PAIR_RADIUS_PX,
    SUBJECT_ID,
    FIRST_IMAGE_ID,
    SECOND_IMAGE_ID,
    candidate_pairs,
    consensus,
    match,
)
from meros.processing.compare_spot_geometry import load
from meros.project import Project


def run(project: Project) -> None:
    samples = load(project)
    first = next(
        sample for sample in samples
        if (sample.subject_id, sample.image_id) == (SUBJECT_ID, FIRST_IMAGE_ID)
    )
    second = next(
        sample for sample in samples
        if (sample.subject_id, sample.image_id) == (SUBJECT_ID, SECOND_IMAGE_ID)
    )

    known = project.metadata.read_subject_spot_correspondences(
        SUBJECT_ID, FIRST_IMAGE_ID, SECOND_IMAGE_ID
    )

    first_tree = cKDTree(first.spots)
    second_tree = cKDTree(second.spots)
    known_indices = []

    for number, pair in enumerate(known, 1):
        distance_a, i = first_tree.query(pair["first"])
        distance_b, j = second_tree.query(pair["second"])

        if max(distance_a, distance_b) <= KNOWN_PAIR_RADIUS_PX:
            known_indices.append((number, int(i), int(j)))

    candidates = {tuple(pair) for pair in candidate_pairs(first.normalized, second.normalized)}
    covered = [(number, i, j) for number, i, j in known_indices if (i, j) in candidates]

    print(f"Known correspondences linked to spot annotations: {len(known_indices)}/{len(known)}")
    print(
        f"Known pairs among top-{CANDIDATES_PER_SPOT} proposals: "
        f"{len(covered)}/{len(known_indices)}"
    )
    print("  Not proposed IDs:", [
        number for number, i, j in known_indices if (i, j) not in candidates
    ])

    # Oracle transforms are fitted exclusively for diagnosis, never fed to the matcher.
    source = np.asarray([first.normalized[i] for _, i, _ in known_indices], dtype=np.float32)
    target = np.asarray([second.normalized[j] for _, _, j in known_indices], dtype=np.float32)

    if len(source) < 3:
        return

    for name, estimator in (
        ("Similarity", cv.estimateAffinePartial2D),
        ("Affine", cv.estimateAffine2D),
    ):
        transform, inliers = estimator(
            source,
            target,
            method=cv.RANSAC,
            ransacReprojThreshold=CONSENSUS_RADIUS,
            maxIters=2000,
            confidence=0.99,
        )

        if transform is None:
            print(f"{name}: no transform")
            continue

        projection = source @ transform[:, :2].T + transform[:, 2]
        residuals = np.linalg.norm(projection - target, axis=1)
        print(
            f"{name} oracle: {int((residuals <= CONSENSUS_RADIUS).sum())}/"
            f"{len(source)} known pairs within {CONSENSUS_RADIUS} eye radii"
        )
        print(
            "  Outlier IDs:",
            [number for (number, _, _), error in zip(known_indices, residuals, strict=True)
             if error > CONSENSUS_RADIUS],
        )

        if name == "Similarity":
            oracle_matches, _ = consensus(
                first.normalized, second.normalized,
                transform[:, :2], transform[:, 2],
            )
            oracle_set = {tuple(pair) for pair in oracle_matches}
            true_set = {(i, j) for _, i, j in known_indices}
            print(
                f"  Full-cloud consensus: {len(oracle_matches)} proposed, "
                f"{len(oracle_set & true_set)} verified"
            )

    selected = {tuple(pair) for pair in match(first.normalized, second.normalized)}
    true_set = {(i, j) for _, i, j in known_indices}
    print(
        f"Matcher result: {len(selected)} proposed, "
        f"{len(selected & true_set)} verified"
    )
    print(
        "\\nOracle uses ground truth only as an upper-bound diagnostic. "
        "It is not a valid reidentification score."
    )


if __name__ == "__main__":
    run(Project.open())
