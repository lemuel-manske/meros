"""Explore local constellations of manually annotated spots.

Compare eye-aligned matching against partial matching of nearby spot groups.
No new annotations or metadata writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from meros.project import Project


EYE_MATCH_RADIUS = 0.24
LOCAL_NEIGHBORS = 6
ANCHOR_CANDIDATES = 2
ORIENTATION_NEIGHBORS = 3
LOCAL_MATCH_RADIUS = 0.20
MIN_LOCAL_MATCHES = 4


@dataclass(frozen=True)
class Sample:
    subject_id: str
    image_id: int
    points: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.subject_id}/{self.image_id}"


def load(project: Project) -> list[Sample]:
    samples = []

    for subject in project.media.list_subjects():
        for photo in subject.images:
            spots = np.asarray(
                project.metadata.read_subject_spots(subject.subject_id, photo.image_id),
                dtype=np.float64,
            ).reshape(-1, 2)

            eye = project.metadata.read_subject_eye(subject.subject_id, photo.image_id)

            if not len(spots) or eye is None:
                continue

            radius = float(np.sqrt(eye.radius_x * eye.radius_y))
            points = (spots - np.asarray(eye.center)) / radius

            samples.append(Sample(subject.subject_id, photo.image_id, points))

    return samples


def mutual_matches(a: np.ndarray, b: np.ndarray, tolerance: float) -> int:
    if not len(a) or not len(b):
        return 0

    distances, forward = cKDTree(b).query(a)
    _, backward = cKDTree(a).query(b)

    return sum(
        distance <= tolerance and int(backward[int(j)]) == i
        for i, (distance, j) in enumerate(zip(distances, forward, strict=True))
    )


def eye_score(a: Sample, b: Sample) -> float:
    return mutual_matches(a.points, b.points, EYE_MATCH_RADIUS) / min(
        len(a.points), len(b.points)
    )


def neighborhoods(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(LOCAL_NEIGHBORS + 1, len(points))
    distances, indices = cKDTree(points).query(points, k=count)

    # Neighbor order is determined by local distance from each anchor.
    offsets = points[indices[:, 1:]] - points[:, None, :]
    lengths = distances[:, 1:]

    return indices[:, 1:], offsets, lengths


def local_score(a: Sample, b: Sample) -> float:
    if min(len(a.points), len(b.points)) < MIN_LOCAL_MATCHES:
        return 0.0

    a_ids, a_offsets, a_lengths = neighborhoods(a.points)
    b_ids, b_offsets, b_lengths = neighborhoods(b.points)

    # Describe an anchor using its ordered neighbor distances. These
    # distances are already normalized by the eye radius.
    candidate_tree = cKDTree(b_lengths)
    _, candidates = candidate_tree.query(
        a_lengths, k=min(ANCHOR_CANDIDATES, len(b.points))
    )
    candidates = np.asarray(candidates).reshape(len(a.points), -1)

    best = 0

    for i, matches in enumerate(candidates):
        for j in matches:
            j = int(j)

            # Try orienting a neighborhood using corresponding neighbor rays.
            for va in a_offsets[i, :ORIENTATION_NEIGHBORS]:
                length_a = float(np.linalg.norm(va))

                if length_a == 0:
                    continue

                for vb in b_offsets[j, :ORIENTATION_NEIGHBORS]:
                    length_b = float(np.linalg.norm(vb))

                    if length_b == 0 or abs(length_a - length_b) > LOCAL_MATCH_RADIUS:
                        continue

                    cosine = float(np.dot(va, vb) / (length_a * length_b))
                    sine = float(np.linalg.det(np.stack((va, vb))) / (length_a * length_b))
                    rotation = np.array([[cosine, -sine], [sine, cosine]])

                    rotated = a_offsets[i] @ rotation.T
                    matched_neighbors = mutual_matches(
                        rotated, b_offsets[j], LOCAL_MATCH_RADIUS
                    )
                    best = max(best, matched_neighbors + 1)

    if best < MIN_LOCAL_MATCHES:
        return 0.0

    # A local agreement is evidence for a subset of spots, not an
    # assertion that the entire head has been aligned.
    return best / (min(LOCAL_NEIGHBORS, len(a.points) - 1, len(b.points) - 1) + 1)


def report(samples: list[Sample], label: str, matcher) -> None:
    scores = np.zeros((len(samples), len(samples)))

    pairs = list(combinations(range(len(samples)), 2))

    for i, j in pairs:
        scores[i, j] = scores[j, i] = matcher(samples[i], samples[j])

    positive = [
        scores[i, j] for i, j in pairs
        if samples[i].subject_id == samples[j].subject_id
    ]
    negative = [
        scores[i, j] for i, j in pairs
        if samples[i].subject_id != samples[j].subject_id
    ]

    print(f"\\n{label}")
    print(f"  same-subject: {len(positive)} pairs, mean={np.mean(positive):.3f}")
    print(f"  different-subject: {len(negative)} pairs, mean={np.mean(negative):.3f}")

    eligible = [
        i for i, sample in enumerate(samples)
        if sum(other.subject_id == sample.subject_id for other in samples) > 1
    ]
    correct = 0

    for i in eligible:
        ranked = sorted(
            (j for j in range(len(samples)) if j != i),
            key=lambda j: (-scores[i, j], samples[j].name),
        )
        winner = ranked[0]
        success = samples[i].subject_id == samples[winner].subject_id
        correct += success

        print(
            f"  {samples[i].name:<18} best={samples[winner].name:<18} "
            f"score={scores[i, winner]:.3f} "
            f"{'correct' if success else 'wrong'}"
        )

    if eligible:
        print(f"  rank-1: {correct}/{len(eligible)} = {correct / len(eligible):.3f}")


def run(project: Project) -> None:
    samples = load(project)
    print(f"Loaded {len(samples)} images with manually annotated eyes and spots.")

    if len(samples) < 2:
        return

    report(samples, "Eye baseline", eye_score)
    report(samples, "Local spot constellations", local_score)

    print(
        "\\nExploratory scores: local and eye scores use different denominators, "
        "so compare discrimination and rank-1, not raw score magnitude."
    )


if __name__ == "__main__":
    run(Project.open())
