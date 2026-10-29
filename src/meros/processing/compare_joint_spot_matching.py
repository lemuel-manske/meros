"""Evaluate joint geometric agreement among spot correspondence candidates."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from meros.processing.compare_spot_geometry import (
    EYE_MATCH_RADIUS,
    eye_score,
    load,
    neighborhood_descriptors,
)
from meros.project import Project


SUBJECT_ID = "luana"
FIRST_IMAGE_ID = 0
SECOND_IMAGE_ID = 1

CANDIDATES_PER_SPOT = 10
HYPOTHESIS_COUNT = 600
CONSENSUS_RADIUS = 0.24  # Eye radii, as in the eye-only baseline.
KNOWN_PAIR_RADIUS_PX = 15.0
RANDOM_SEED = 17
DESCRIPTOR_WEIGHT = 0.45
TRANSFORM_PENALTY = 0.15


def candidate_pairs(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    first = neighborhood_descriptors(source)
    second = neighborhood_descriptors(target)

    _, neighbors = cKDTree(second).query(
        first, k=min(CANDIDATES_PER_SPOT, len(target))
    )

    neighbors = np.asarray(neighbors).reshape(len(source), -1)

    return np.array(
        [(i, int(j)) for i, row in enumerate(neighbors) for j in row],
        dtype=int,
    )


def consensus(
    first: np.ndarray, second: np.ndarray, rotation: np.ndarray, offset: np.ndarray
) -> tuple[np.ndarray, float]:
    transformed = first @ rotation.T + offset
    distances, forward = cKDTree(second).query(transformed)
    _, backward = cKDTree(transformed).query(second)

    matches = np.array(
        [
            (i, int(j))
            for i, (distance, j) in enumerate(zip(distances, forward, strict=True))
            if distance <= CONSENSUS_RADIUS and int(backward[int(j)]) == i
        ],
        dtype=int,
    ).reshape(-1, 2)

    return matches, float(distances[matches[:, 0]].sum()) if len(matches) else 0.0


def match(first: np.ndarray, second: np.ndarray, weighted: bool = False) -> np.ndarray:
    if min(len(first), len(second)) < 3:
        return np.empty((0, 2), dtype=int)

    candidates = candidate_pairs(first, second)
    rng = np.random.default_rng(RANDOM_SEED)

    best = np.empty((0, 2), dtype=int)
    best_error = float("inf")
    best_quality = float("-inf")

    descriptor_a = neighborhood_descriptors(first) if weighted else None
    descriptor_b = neighborhood_descriptors(second) if weighted else None

    # Two distinct tentative correspondences define a similarity transform.
    # Score each hypothesis against ALL observed spots, not just the proposals.
    for _ in range(HYPOTHESIS_COUNT):
        a, b = candidates[rng.choice(len(candidates), size=2, replace=False)]

        if a[0] == b[0] or a[1] == b[1]:
            continue

        source_vector = first[b[0]] - first[a[0]]
        target_vector = second[b[1]] - second[a[1]]

        source_length = float(np.linalg.norm(source_vector))
        target_length = float(np.linalg.norm(target_vector))

        if source_length < 0.25 or target_length < 0.25:
            continue

        # The eye already supplies an approximate scale.
        scale = target_length / source_length

        if not 0.65 <= scale <= 1.5:
            continue

        cosine = float(source_vector @ target_vector / (source_length * target_length))
        sine = float(np.linalg.det(np.stack((source_vector, target_vector))) /
                     (source_length * target_length))
        transform = scale * np.array([[cosine, -sine], [sine, cosine]])
        translation = second[a[1]] - first[a[0]] @ transform.T

        matched, error = consensus(first, second, transform, translation)

        if weighted:
            descriptor_distance = (
                np.linalg.norm(
                    descriptor_a[matched[:, 0]] - descriptor_b[matched[:, 1]], axis=1
                )
                if len(matched)
                else np.empty(0)
            )
            weights = np.exp(-DESCRIPTOR_WEIGHT * descriptor_distance)
            transform_cost = TRANSFORM_PENALTY * (
                abs(float(np.log(scale)))
                + abs(float(np.arctan2(sine, cosine)))
                + float(np.linalg.norm(translation))
            )
            quality = float(weights.sum()) - transform_cost

            if quality > best_quality:
                best, best_quality = matched, quality

        elif len(matched) > len(best) or (
            len(matched) == len(best) and error < best_error
        ):
            best, best_error = matched, error

    return best


def run(project: Project) -> None:
    samples = load(project)
    print(f"Loaded {len(samples)} images with annotated spots and eyes.")

    if len(samples) < 2:
        return

    eye_scores = np.zeros((len(samples), len(samples)))
    joint_scores = np.zeros_like(eye_scores)
    selected_pairs = {}
    weighted_pairs = {}
    weighted_scores = np.zeros_like(eye_scores)

    for i, j in combinations(range(len(samples)), 2):
        eye_scores[i, j] = eye_scores[j, i] = eye_score(samples[i], samples[j], False)

        pairs = match(samples[i].normalized, samples[j].normalized)
        selected_pairs[i, j] = pairs

        refined = match(samples[i].normalized, samples[j].normalized, weighted=True)
        weighted_pairs[i, j] = refined
        weighted_scores[i, j] = weighted_scores[j, i] = (
            len(refined) / min(len(samples[i].spots), len(samples[j].spots))
        )

        joint_scores[i, j] = joint_scores[j, i] = (
            len(pairs) / min(len(samples[i].spots), len(samples[j].spots))
        )

    eligible = [
        i for i, sample in enumerate(samples)
        if sum(other.subject_id == sample.subject_id for other in samples) > 1
    ]

    for label, scores in (
        ("Eye-only baseline", eye_scores),
        ("Joint geometric consensus", joint_scores),
        ("Weighted descriptor + eye prior", weighted_scores),
    ):
        positives = [
            scores[i, j] for i, j in combinations(range(len(samples)), 2)
            if samples[i].subject_id == samples[j].subject_id
        ]
        negatives = [
            scores[i, j] for i, j in combinations(range(len(samples)), 2)
            if samples[i].subject_id != samples[j].subject_id
        ]

        correct = 0

        print(f"\\n{label}")
        print(
            f"  positive mean={np.mean(positives):.3f}, "
            f"negative mean={np.mean(negatives):.3f}"
        )

        for i in eligible:
            winner = sorted(
                (j for j in range(len(samples)) if j != i),
                key=lambda j: (-scores[i, j], samples[j].name),
            )[0]
            correct += samples[i].subject_id == samples[winner].subject_id

        print(f"  rank-1: {correct}/{len(eligible)}")

    first_index = next(
        i for i, sample in enumerate(samples)
        if (sample.subject_id, sample.image_id) == (SUBJECT_ID, FIRST_IMAGE_ID)
    )
    second_index = next(
        i for i, sample in enumerate(samples)
        if (sample.subject_id, sample.image_id) == (SUBJECT_ID, SECOND_IMAGE_ID)
    )

    i, j = sorted((first_index, second_index))
    proposed = selected_pairs[i, j]

    known = project.metadata.read_subject_spot_correspondences(
        SUBJECT_ID, FIRST_IMAGE_ID, SECOND_IMAGE_ID
    )
    first_coords = samples[first_index].spots
    second_coords = samples[second_index].spots

    verified = []
    for pair in known:
        left = np.asarray(pair["first"])
        right = np.asarray(pair["second"])
        distance_a, index_a = cKDTree(first_coords).query(left)
        distance_b, index_b = cKDTree(second_coords).query(right)

        if max(distance_a, distance_b) <= KNOWN_PAIR_RADIUS_PX:
            verified.append((int(index_a), int(index_b)))

    oriented = proposed if first_index == i else proposed[:, ::-1]
    proposed_set = {tuple(row) for row in oriented}
    verified_set = set(verified)

    confirmed = proposed_set & verified_set
    conflicts = {
        pair
        for pair in proposed_set - verified_set
        if any(
            (pair[0] == ground[0] and pair[1] != ground[1])
            or (pair[1] == ground[1] and pair[0] != ground[0])
            for ground in verified_set
        )
    }
    unverified = proposed_set - confirmed - conflicts

    print(f"\\nKnown {SUBJECT_ID} correspondences (not used to fit the matcher):")
    print(f"  eligible verified pairs: {len(verified_set)}/{len(known)}")
    print(f"  verified pairs retrieved: {len(confirmed)}/{len(verified_set)}")
    print(f"  proposed pairs: {len(proposed_set)}")
    print(f"  confirmed proposals: {len(confirmed)}")
    print(f"  conflicting proposals: {len(conflicts)}")
    print(f"  unverified proposals: {len(unverified)}")

    refined = weighted_pairs[i, j]
    refined_oriented = refined if first_index == i else refined[:, ::-1]
    refined_set = {tuple(row) for row in refined_oriented}
    print(
        f"  weighted proposals: {len(refined_set)}, "
        f"verified: {len(refined_set & verified_set)}"
    )
    print(
        "  Note: unverified does not mean incorrect; conflicts assume "
        "one-to-one spot identity."
    )
    print(
        "\\nExploratory: a single annotated positive pair cannot establish "
        "generalization to other individuals."
    )


if __name__ == "__main__":
    run(Project.open())
