"""Explore subject reidentification with manually annotated spots and eyes.

No detector training: compare eye-aligned, eye-rotation-search, and
geometry-only RANSAC matching on independent subject photographs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2 as cv
import numpy as np
from scipy.spatial import cKDTree

from meros.project import Project


# Adjust in the script; no CLI flags.
EYE_MATCH_RADIUS = 0.24  # Eye radii in normalized coordinates.
ROTATIONS_DEGREES = tuple(range(-30, 31, 5))
DESCRIPTOR_NEIGHBORS = 6
DESCRIPTOR_RATIO = 0.85
RANSAC_REPROJECTION_PX = 5.0
RANSAC_ITERATIONS = 2000


@dataclass(frozen=True)
class Sample:
    subject_id: str
    image_id: int
    spots: np.ndarray
    normalized: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.subject_id}/{self.image_id}"


def load(project: Project) -> list[Sample]:
    samples = []

    for subject in project.media.list_subjects():
        for photo in subject.images:
            spots = np.asarray(
                project.metadata.read_subject_spots(subject.subject_id, photo.image_id),
                dtype=np.float32,
            ).reshape(-1, 2)
            eye = project.metadata.read_subject_eye(subject.subject_id, photo.image_id)

            if len(spots) == 0 or eye is None:
                continue

            radius = float(np.sqrt(eye.radius_x * eye.radius_y))
            normalized = (spots - np.asarray(eye.center, dtype=np.float32)) / radius
            samples.append(Sample(subject.subject_id, photo.image_id, spots, normalized))

    return samples


def mutual_matches(left: np.ndarray, right: np.ndarray, tolerance: float) -> int:
    """Count unique, bidirectional point neighbors within the chosen tolerance."""
    if not len(left) or not len(right):
        return 0

    left_dist, left_idx = cKDTree(right).query(left)
    _, right_idx = cKDTree(left).query(right)

    return sum(
        float(distance) <= tolerance and int(right_idx[int(index)]) == i
        for i, (distance, index) in enumerate(zip(left_dist, left_idx, strict=True))
    )


def eye_score(left: Sample, right: Sample, rotate: bool) -> float:
    angles = ROTATIONS_DEGREES if rotate else (0,)
    best = 0

    for degrees in angles:
        angle = np.deg2rad(degrees)
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        matches = mutual_matches(
            left.normalized @ matrix.T, right.normalized, EYE_MATCH_RADIUS
        )
        best = max(best, matches)

    return best / min(len(left.spots), len(right.spots))


def neighborhood_descriptors(points: np.ndarray) -> np.ndarray:
    """Scale-free local distance patterns used to propose point correspondences."""
    neighbors = min(len(points) - 1, DESCRIPTOR_NEIGHBORS)

    if neighbors < 2:
        return np.empty((0, DESCRIPTOR_NEIGHBORS), dtype=np.float32)

    distances, _ = cKDTree(points).query(points, k=neighbors + 1)
    local = distances[:, 1:]
    scale = np.maximum(np.median(local, axis=1, keepdims=True), 1e-6)
    features = local / scale
    # Pad short neighborhoods so both photographs have equal feature width.
    return np.pad(
        features,
        ((0, 0), (0, DESCRIPTOR_NEIGHBORS - neighbors)),
        mode="edge",
    ).astype(np.float32)


def geometry_score(left: Sample, right: Sample) -> float:
    if min(len(left.spots), len(right.spots)) < 3:
        return 0.0

    first = neighborhood_descriptors(left.spots)
    second = neighborhood_descriptors(right.spots)
    distances, indices = cKDTree(second).query(first, k=2)

    # Candidate matches are mutually preferred in descriptor space.
    best_back = cKDTree(first).query(second)[1]
    pairs = [
        (i, int(j))
        for i, (j, (best, alternate)) in enumerate(
            zip(indices[:, 0], distances, strict=True)
        )
        if best < DESCRIPTOR_RATIO * max(float(alternate), 1e-8)
        and int(best_back[int(j)]) == i
    ]

    if len(pairs) < 3:
        return 0.0

    source = np.asarray([left.spots[i] for i, _ in pairs], dtype=np.float32)
    target = np.asarray([right.spots[j] for _, j in pairs], dtype=np.float32)

    # Threshold scales with the target image's typical annotated-point spacing.
    spacing = cKDTree(right.spots).query(right.spots, k=2)[0][:, 1]
    threshold = min(
        RANSAC_REPROJECTION_PX,
        max(2.0, float(np.median(spacing)) * 0.25),
    )
    transform, inliers = cv.estimateAffinePartial2D(
        source,
        target,
        method=cv.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=RANSAC_ITERATIONS,
        confidence=0.99,
    )

    if transform is None or inliers is None:
        return 0.0

    # Score includes consensus size, not just the inlier fraction among proposals.
    return float(inliers.sum()) / min(len(left.spots), len(right.spots))


def summarize(
    samples: list[Sample],
    label: str,
    scores: np.ndarray,
) -> None:
    pairs = list(combinations(range(len(samples)), 2))
    positives = [
        scores[a, b] for a, b in pairs
        if samples[a].subject_id == samples[b].subject_id
    ]
    negatives = [
        scores[a, b] for a, b in pairs
        if samples[a].subject_id != samples[b].subject_id
    ]

    print(f"\n{label}")
    print(
        f"  same-subject pairs={len(positives)} mean={np.mean(positives):.3f}"
        if positives else "  no same-subject pairs"
    )
    print(
        f"  different-subject pairs={len(negatives)} mean={np.mean(negatives):.3f}"
        if negatives else "  no different-subject pairs"
    )

    eligible = [
        i for i, sample in enumerate(samples)
        if sum(other.subject_id == sample.subject_id for other in samples) > 1
    ]
    correct = 0

    for i in eligible:
        candidates = [j for j in range(len(samples)) if j != i]
        ranked = sorted(candidates, key=lambda j: (-scores[i, j], samples[j].name))
        best = ranked[0]
        is_correct = samples[best].subject_id == samples[i].subject_id
        correct += is_correct
        print(
            f"  {samples[i].name:<18} best={samples[best].name:<18} "
            f"score={scores[i, best]:.3f} "
            f"{'correct' if is_correct else 'wrong'}"
        )

    if eligible:
        print(f"  rank-1: {correct}/{len(eligible)} = {correct / len(eligible):.3f}")


def run(project: Project) -> None:
    samples = load(project)
    print(f"Loaded {len(samples)} images with nonempty spots and an eye ellipse.")

    if len(samples) < 2:
        print("Need at least two annotated images.")
        return

    methods = {
        "Eye center + scale": lambda a, b: eye_score(a, b, False),
        "Eye center + scale + rotation search": lambda a, b: eye_score(a, b, True),
        "Geometry-only descriptor + RANSAC": geometry_score,
    }

    for label, matcher in methods.items():
        scores = np.zeros((len(samples), len(samples)), dtype=np.float32)

        for i, j in combinations(range(len(samples)), 2):
            score = matcher(samples[i], samples[j])
            scores[i, j] = score
            scores[j, i] = score

        summarize(samples, label, scores)

    print(
        "\nExploratory evaluation: annotations are incomplete, candidate scores "
        "are uncalibrated, and parameters were not selected on held-out subjects."
    )


if __name__ == "__main__":
    run(Project.open())
