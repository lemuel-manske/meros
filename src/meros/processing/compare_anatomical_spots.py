from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt

import numpy as np
from scipy.spatial.distance import cdist

from meros.project import Project


HORIZONTAL_SCALES = (0.83, 1.0, 1.2)
MATCH_RADIUS_EYE = 0.6
MANUAL_LINK_RADIUS_PX = 15.0


@dataclass(frozen=True)
class Sample:
    subject_id: str
    image_id: int
    spots: np.ndarray
    normalized_spots: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.subject_id}/{self.image_id}"


@dataclass(frozen=True)
class Match:
    score: float
    pairs: tuple[tuple[int, int], ...]
    scale: float
    reversed_direction: bool


def read_spots(project: Project, subject_id: str, image_id: int) -> np.ndarray:
    return np.asarray(
        project.metadata.read_subject_spots(subject_id, image_id),
        dtype=float,
    ).reshape(-1, 2)


def normalize_spots(
    spots: np.ndarray,
    eye_center: tuple[float, float],
    eye_radius_x: float,
    eye_radius_y: float,
    mouth: tuple[float, float],
) -> np.ndarray:
    eye = np.asarray(eye_center, dtype=float)
    axis = np.asarray(mouth, dtype=float) - eye
    axis /= np.linalg.norm(axis)

    transverse = np.array((axis[1], -axis[0]))
    radius = (eye_radius_x + eye_radius_y) / 2

    relative = spots - eye
    return np.column_stack((relative @ transverse, relative @ axis)) / radius


def directional_match(
    source: np.ndarray,
    target: np.ndarray,
    scale: float,
) -> tuple[tuple[int, int], ...]:
    transformed = source.copy()
    transformed[:, 0] *= scale

    distances = cdist(transformed, target)
    source_indices, target_indices = np.where(distances <= MATCH_RADIUS_EYE)
    order = np.argsort(
        distances[source_indices, target_indices],
        kind="stable",
    )

    used_source: set[int] = set()
    used_target: set[int] = set()
    matches = []

    for position in order:
        source_id = int(source_indices[position])
        target_id = int(target_indices[position])

        if source_id in used_source or target_id in used_target:
            continue

        used_source.add(source_id)
        used_target.add(target_id)
        matches.append((source_id, target_id))

    return tuple(matches)


def compare(left: Sample, right: Sample) -> Match:
    best = Match(0.0, (), 1.0, False)
    normalization = sqrt(len(left.spots) * len(right.spots))

    for reversed_direction, (source, target) in (
        (False, (left.normalized_spots, right.normalized_spots)),
        (True, (right.normalized_spots, left.normalized_spots)),
    ):
        for scale in HORIZONTAL_SCALES:
            pairs = directional_match(source, target, scale)
            score = len(pairs) / normalization

            if score > best.score:
                if reversed_direction:
                    pairs = tuple((j, i) for i, j in pairs)

                best = Match(score, pairs, scale, reversed_direction)

    return best


def evaluate_subject(
    project: Project,
    subject_id: str,
    samples: list[Sample],
    results: dict[tuple[int, int], Match]
) -> None:
    subject_samples = [sample for sample in samples if sample.subject_id == subject_id]

    if len(subject_samples) < 2:
        return

    print(f"Subject {subject_id} ({len(subject_samples)} images):")

    for i, j in combinations(range(len(subject_samples)), 2):
        result = results[(min(i, j), max(i, j))]
        print(
            f"  {subject_samples[i].label} vs {subject_samples[j].label}: "
            f"{len(result.pairs)} matches, score={result.score:.3f}, "
            f"scale={result.scale:.3f}, reversed={result.reversed_direction}",
        )


def evaluate_luana(
    project: Project,
    subject: str,
    samples: list[Sample],
    results: dict[tuple[int, int], Match],
) -> None:
    labels = [sample.label for sample in samples]

    i, j = labels.index("luana/0"), labels.index("luana/1")
    first, second = samples[i], samples[j]

    manual = project.metadata.read_subject_spot_correspondences("luana", 0, 1)
    verified_pairs = set()

    for annotation in manual:
        distances_a = np.linalg.norm(first.spots - annotation["first"], axis=1)
        distances_b = np.linalg.norm(second.spots - annotation["second"], axis=1)
        index_a = int(np.argmin(distances_a))
        index_b = int(np.argmin(distances_b))

        if max(distances_a[index_a], distances_b[index_b]) <= MANUAL_LINK_RADIUS_PX:
            verified_pairs.add((index_a, index_b))

    found = len(verified_pairs.intersection(results[(i, j)].pairs))

    print(f"{subject} verified correspondences recovered: {found}/{len(verified_pairs)}")


def run(project: Project) -> None:
    samples = []

    for subject in project.media.list_subjects():
        for photo in subject.images:
            spots = read_spots(project, subject.subject_id, photo.image_id)
            eye = project.metadata.read_subject_eye(subject.subject_id, photo.image_id)
            mouth = project.metadata.read_subject_mouth(subject.subject_id, photo.image_id)

            if not len(spots) or eye is None or mouth is None:
                continue

            samples.append(
                Sample(
                    subject_id=subject.subject_id,
                    image_id=photo.image_id,
                    spots=spots,
                    normalized_spots=normalize_spots(
                        spots,
                        eye.center,
                        eye.radius_x,
                        eye.radius_y,
                        mouth,
                    ),
                )
            )

    results = {}
    positives = []
    negatives = []

    for i, j in combinations(range(len(samples)), 2):
        result = compare(samples[i], samples[j])
        results[(i, j)] = result

        if samples[i].subject_id == samples[j].subject_id:
            positives.append(result.score)
        else:
            negatives.append(result.score)

    print(f"Images: {len(samples)}")
    print(f"Positive pairs: {len(positives)}, mean={np.mean(positives):.3f}")
    print(f"Negative pairs: {len(negatives)}, mean={np.mean(negatives):.3f}")

    correct = 0
    queries = 0

    for i, source in enumerate(samples):
        if sum(other.subject_id == source.subject_id for other in samples) < 2:
            continue

        queries += 1
        candidates = [
            (
                results[(min(i, j), max(i, j))].score,
                j,
            )
            for j in range(len(samples))
            if i != j
        ]
        score, j = max(candidates, key=lambda candidate: candidate[0])
        correct += samples[j].subject_id == source.subject_id

        print(f"  {source.label} best={samples[j].label} score={score:.3f}")

    print(f"Rank-1: {correct}/{queries}")

    evaluate_luana(project, samples, results)


if __name__ == "__main__":
    run(Project.open())
