"""Compare eye-only vs eye-to-mouth anatomical spot alignment.

Uses manually annotated points only. No model fitting or metadata writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from meros.project import Project


MATCH_TOLERANCE = 0.12  # Fraction of the eye-to-mouth distance.
SHOW_LUANA_OVERLAY = True
OVERLAY_SUBJECT = "luana"
OVERLAY_IMAGES = (0, 1)


@dataclass(frozen=True)
class Sample:
    subject_id: str
    image_id: int
    eye: np.ndarray
    radius: float
    mouth: np.ndarray
    spots: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.subject_id}/{self.image_id}"

    def points(self, mode: str) -> np.ndarray:
        delta = self.mouth - self.eye
        distance = float(np.linalg.norm(delta))

        if mode == "eye":
            return (self.spots - self.eye) / self.radius

        direction = delta / distance
        perpendicular = np.array([-direction[1], direction[0]])
        centered = self.spots - self.eye

        return np.column_stack(
            (centered @ direction, centered @ perpendicular)
        ) / distance


def load(project: Project) -> list[Sample]:
    samples = []

    for subject in project.media.list_subjects():
        for photo in subject.images:
            spots = np.asarray(
                project.metadata.read_subject_spots(subject.subject_id, photo.image_id),
                dtype=np.float64,
            ).reshape(-1, 2)
            eye = project.metadata.read_subject_eye(subject.subject_id, photo.image_id)
            mouth = project.metadata.read_subject_mouth(subject.subject_id, photo.image_id)

            if not len(spots) or eye is None or mouth is None:
                continue

            samples.append(
                Sample(
                    subject.subject_id,
                    photo.image_id,
                    np.asarray(eye.center, dtype=np.float64),
                    float(np.sqrt(eye.radius_x * eye.radius_y)),
                    np.asarray(mouth, dtype=np.float64),
                    spots,
                )
            )

    return samples


def mutual_count(a: np.ndarray, b: np.ndarray, tolerance: float) -> int:
    if not len(a) or not len(b):
        return 0

    distances, forward = cKDTree(b).query(a)
    _, backward = cKDTree(a).query(b)

    return sum(
        distance <= tolerance and int(backward[int(j)]) == i
        for i, (distance, j) in enumerate(zip(distances, forward, strict=True))
    )


def report(samples: list[Sample], mode: str) -> None:
    coords = [sample.points(mode) for sample in samples]
    score = np.zeros((len(samples), len(samples)))

    for i, j in combinations(range(len(samples)), 2):
        # The eye-only baseline uses the same physical pixel tolerance,
        # expressed in local eye-radius coordinates for both images.
        if mode == "eye":
            normalized_distance = (
                np.linalg.norm(samples[i].mouth - samples[i].eye)
                + np.linalg.norm(samples[j].mouth - samples[j].eye)
            ) / (samples[i].radius + samples[j].radius)
            tolerance = MATCH_TOLERANCE * normalized_distance
        else:
            tolerance = MATCH_TOLERANCE

        matched = mutual_count(coords[i], coords[j], tolerance)
        score[i, j] = score[j, i] = matched / min(len(coords[i]), len(coords[j]))

    positive = [
        score[i, j]
        for i, j in combinations(range(len(samples)), 2)
        if samples[i].subject_id == samples[j].subject_id
    ]
    negative = [
        score[i, j]
        for i, j in combinations(range(len(samples)), 2)
        if samples[i].subject_id != samples[j].subject_id
    ]

    print(f"\n{mode}: positive_pairs={len(positive)} negative_pairs={len(negative)}")
    print(f"  mean positive={np.mean(positive):.3f} negative={np.mean(negative):.3f}")

    eligible = [
        i for i, item in enumerate(samples)
        if sum(other.subject_id == item.subject_id for other in samples) > 1
    ]
    correct = 0

    for i in eligible:
        ranking = sorted(
            (j for j in range(len(samples)) if j != i),
            key=lambda j: (-score[i, j], samples[j].name),
        )
        winner = ranking[0]
        ok = samples[i].subject_id == samples[winner].subject_id
        correct += ok
        print(
            f"  {samples[i].name:<16} best={samples[winner].name:<16} "
            f"score={score[i, winner]:.3f} {'correct' if ok else 'wrong'}"
        )

    if eligible:
        print(f"  rank-1: {correct}/{len(eligible)} = {correct / len(eligible):.3f}")


def show_overlay(samples: list[Sample]) -> None:
    selected = [
        item
        for item in samples
        if item.subject_id == OVERLAY_SUBJECT and item.image_id in OVERLAY_IMAGES
    ]

    if len(selected) != 2:
        return

    figure, axes = plt.subplots(1, 2, figsize=(13, 6))

    for axis, mode in zip(axes, ("eye", "eye+mouth"), strict=True):
        for sample, color in zip(selected, ("tab:blue", "tab:orange"), strict=True):
            points = sample.points(mode)
            axis.scatter(points[:, 0], points[:, 1], s=16, alpha=0.7, label=sample.name, c=color)

        axis.scatter([0], [0], c="black", marker="+", s=100)
        axis.set_title(mode)
        axis.set_aspect("equal")
        axis.legend()

    figure.tight_layout()
    plt.show()
    plt.close(figure)


def run(project: Project) -> None:
    samples = load(project)
    print(f"Loaded {len(samples)} annotated subject images with eye, mouth and spots.")

    if len(samples) < 2:
        return

    for mode in ("eye", "eye+mouth"):
        report(samples, mode)

    if SHOW_LUANA_OVERLAY:
        show_overlay(samples)

    print(
        "\nExploratory matching only; a two-landmark similarity transform "
        "cannot eliminate three-dimensional perspective deformation."
    )


if __name__ == "__main__":
    run(Project.open())
