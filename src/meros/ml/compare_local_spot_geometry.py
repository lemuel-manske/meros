"""
Compare neighborhood geometry against proximity-only spot matching.

Only spots and anatomical anchors are matcher inputs. Verified correspondences
are used solely for evaluation after the matching step.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist

from meros.ml.match_spot_patterns import (
    SpotMatch,
    initial_similarity,
    match_spots,
    project_points,
    verified_pairs,
)
from meros.project import Project


RADIUS_RATIO = 0.12
NEIGHBORS = 5
DESCRIPTOR_LIMIT = 0.18

POSITIVE_PAIRS = (
    (("luana", 0), ("luana", 1)),
    (("rafael", 0), ("rafael", 1)),
    (("rogerio", 0), ("rogerio", 1)),
    (("rogerio", 0), ("rogerio", 9)),
    (("yam", 0), ("yam", 1)),
)
NEGATIVE_PAIRS = (
    (("luana", 0), ("rafael", 1)),
    (("luana", 0), ("rogerio", 1)),
    (("luana", 0), ("yam", 1)),
    (("rafael", 0), ("rogerio", 1)),
    (("rafael", 0), ("yam", 1)),
    (("rogerio", 0), ("yam", 1)),
)


def descriptors(points: np.ndarray, scale: float) -> np.ndarray:
    """
    Describe each spot using the distances to its nearest neighbors.
    """
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)

    return np.sort(distances, axis=1)[:, :NEIGHBORS] / scale


def associate(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    source_scale: float,
    target_scale: float,
) -> tuple[tuple[int, int], ...]:
    """
    Associate spots with both projected-position and neighborhood constraints.
    """
    radius = RADIUS_RATIO * target_scale
    position_distances = cdist(project_points(source, matrix), target)
    source_descriptors = descriptors(source, source_scale)
    target_descriptors = descriptors(target, target_scale)
    count = min(source_descriptors.shape[1], target_descriptors.shape[1])

    if count == 0:
        return ()

    neighborhood_distances = cdist(
        source_descriptors[:, :count],
        target_descriptors[:, :count],
    ) / np.sqrt(count)

    eligible = (
        (position_distances <= radius)
        & (neighborhood_distances <= DESCRIPTOR_LIMIT)
    )
    first_indices, second_indices = np.where(eligible)
    scores = (
        position_distances[first_indices, second_indices] / radius
        + neighborhood_distances[first_indices, second_indices] / DESCRIPTOR_LIMIT
    )
    order = np.argsort(scores, kind="stable")
    used_first: set[int] = set()
    used_second: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for index in order:
        first = int(first_indices[index])
        second = int(second_indices[index])

        if first in used_first or second in used_second:
            continue

        pairs.append((first, second))
        used_first.add(first)
        used_second.add(second)

    return tuple(pairs)


def match_local(
    source: np.ndarray,
    target: np.ndarray,
    source_eye: tuple[float, float],
    source_mouth: tuple[float, float],
    target_eye: tuple[float, float],
    target_mouth: tuple[float, float],
) -> SpotMatch:
    """
    Use neighborhood-constrained proposals to estimate an affine refinement.
    """
    source_scale = float(np.linalg.norm(np.asarray(source_mouth) - source_eye))
    target_scale = float(np.linalg.norm(np.asarray(target_mouth) - target_eye))
    radius = RADIUS_RATIO * target_scale
    matrix = initial_similarity(source_eye, source_mouth, target_eye, target_mouth)
    pairs = associate(source, target, matrix, source_scale, target_scale)

    if len(pairs) >= 3:
        first_indices, second_indices = zip(*pairs, strict=True)

        try:
            affine, inliers = cv.estimateAffine2D(
                source[list(first_indices)],
                target[list(second_indices)],
                method=cv.RANSAC,
                ransacReprojThreshold=radius,
            )
        except cv.error:
            affine, inliers = None, None

        if affine is not None and inliers is not None and int(inliers.sum()) >= 3:
            candidate = np.vstack((affine, [0.0, 0.0, 1.0]))
            candidate_pairs = associate(
                source, target, candidate, source_scale, target_scale
            )

            if len(candidate_pairs) >= len(pairs):
                matrix = candidate
                pairs = candidate_pairs

    projected = project_points(source, matrix)
    errors = tuple(float(np.linalg.norm(projected[i] - target[j])) for i, j in pairs)

    return SpotMatch(matrix=matrix, pairs=pairs, errors=errors)


def evaluate(
    project: Project,
    source_label: tuple[str, int],
    target_label: tuple[str, int],
) -> None:
    """
    Report comparable results on positive and negative image pairs.
    """
    source_eye = project.metadata.read_subject_eye(*source_label)
    target_eye = project.metadata.read_subject_eye(*target_label)
    source_mouth = project.metadata.read_subject_mouth(*source_label)
    target_mouth = project.metadata.read_subject_mouth(*target_label)

    if (
        source_eye is None
        or target_eye is None
        or source_mouth is None
        or target_mouth is None
    ):
        print(f"SKIP {source_label} vs {target_label}: missing anatomical landmarks")
        return

    source = np.asarray(project.metadata.read_subject_spots(*source_label), dtype=float)
    target = np.asarray(project.metadata.read_subject_spots(*target_label), dtype=float)

    if len(source) < 3 or len(target) < 3:
        print(f"SKIP {source_label} vs {target_label}: insufficient spots")
        return

    proximity = match_spots(
        source, target, source_eye.center, source_mouth,
        target_eye.center, target_mouth, radius_ratio=RADIUS_RATIO,
    )
    local = match_local(
        source, target, source_eye.center, source_mouth,
        target_eye.center, target_mouth,
    )
    same = source_label[0] == target_label[0]
    verified = (
        verified_pairs(project, source_label, target_label, source, target)
        if same
        else set()
    )

    print(f"\n{source_label} vs {target_label} ({'same' if same else 'different'})")
    print("method | proposed | verified | conflicts | unverified | true_med_px")

    for name, result in (("proximity", proximity), ("local", local)):
        proposed = set(result.pairs)
        recovered = proposed & verified
        first_known = {i for i, _ in verified}
        second_known = {j for _, j in verified}
        conflicts = {
            (i, j)
            for i, j in proposed - verified
            if i in first_known or j in second_known
        }
        unverified = len(proposed) - len(recovered) - len(conflicts)

        if verified:
            indices = sorted(verified)
            projected = project_points(source[[i for i, _ in indices]], result.matrix)
            expected = target[[j for _, j in indices]]
            errors = np.linalg.norm(projected - expected, axis=1)
            verified_text = f"{len(recovered)}/{len(verified)}"
            conflict_text = str(len(conflicts))
            true_error_text = f"{np.median(errors):.1f}"
        else:
            verified_text = "n/a"
            conflict_text = "n/a"
            true_error_text = "n/a"

        print(
            f"{name} | {len(proposed)} | {verified_text} | "
            f"{conflict_text} | {unverified} | {true_error_text}"
        )


def run(project: Project) -> None:
    """
    Compare both matchers on fixed same-individual and different-individual pairs.
    """
    for source_label, target_label in (*POSITIVE_PAIRS, *NEGATIVE_PAIRS):
        evaluate(project, source_label, target_label)


if __name__ == "__main__":
    run(Project.open())
