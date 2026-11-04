"""
Shared geometry sandbox utilities for transform demos and spot registration.

Spot coordinates and error thresholds are in original image pixels. Known spot
correspondences are evaluation labels, not evidence of identity for other fish.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from meros.ml.compare_spot_transforms import project_points
from meros.project import Project


COLOR_SOURCE = "#33ccdd"
COLOR_TARGET = "#ffcc59"


@dataclass(frozen=True)
class SpotExample:
    source_image: np.ndarray
    target_image: np.ndarray
    source_spots: np.ndarray
    target_spots: np.ndarray
    matched_source: np.ndarray
    matched_target: np.ndarray


def load_example(project: Project, subject: str, source_id: int, target_id: int) -> SpotExample:
    """
    Load original photos, all annotated spots, and verified matching pairs.
    """
    source_image = project.media.read_subject_image(subject, source_id)
    target_image = project.media.read_subject_image(subject, target_id)
    source_spots = np.asarray(project.metadata.read_subject_spots(subject, source_id), dtype=float)
    target_spots = np.asarray(project.metadata.read_subject_spots(subject, target_id), dtype=float)
    pairs = project.metadata.read_subject_spot_correspondences(subject, source_id, target_id)

    if not pairs:
        raise ValueError(
            f"Missing verified spot correspondences for {subject}/{source_id}-{target_id}"
        )

    matched_source = np.asarray([pair["first"] for pair in pairs], dtype=float)
    matched_target = np.asarray([pair["second"] for pair in pairs], dtype=float)

    return SpotExample(
        source_image, target_image, source_spots, target_spots, matched_source, matched_target
    )


def errors(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.linalg.norm(predicted - expected, axis=1)


def summarize(name: str, predicted: np.ndarray, example: SpotExample) -> None:
    """
    Report verified-pair error and nearest annotated spot identity accuracy.
    """
    values = errors(predicted, example.matched_target)
    true_indices = cdist(example.matched_target, example.target_spots).argmin(axis=1)
    predicted_indices = cdist(predicted, example.target_spots).argmin(axis=1)
    correct = int(np.sum(true_indices == predicted_indices))

    print(
        f"{name:18s} median={np.median(values):6.1f}px "
        f"mean={np.mean(values):6.1f}px "
        f"within 10px={int(np.sum(values <= 10)):2d}/{len(values)} "
        f"nearest-spot identity={correct:2d}/{len(values)}"
    )


def show_predictions(
    example: SpotExample,
    predicted: np.ndarray,
    title: str,
    *,
    transformed_spots: np.ndarray | None = None,
) -> None:
    """
    Show ground-truth correspondences and residuals on the original photographs.
    """
    figure, axes = plt.subplots(1, 3, figsize=(18, 7))

    axes[0].imshow(cv.cvtColor(example.source_image, cv.COLOR_BGR2RGB))
    axes[0].scatter(
        example.source_spots[:, 0],
        example.source_spots[:, 1],
        s=10,
        color=COLOR_SOURCE,
        alpha=0.5,
    )
    axes[0].scatter(
        example.matched_source[:, 0],
        example.matched_source[:, 1],
        s=32,
        facecolors="none",
        edgecolors="white",
    )
    axes[0].set_title("Source: annotated spots and known pairs")

    axes[1].imshow(cv.cvtColor(example.target_image, cv.COLOR_BGR2RGB))
    axes[1].scatter(
        example.target_spots[:, 0],
        example.target_spots[:, 1],
        s=12,
        color=COLOR_TARGET,
    )
    axes[1].set_title("Target: all annotated spots")

    axes[2].imshow(cv.cvtColor(example.target_image, cv.COLOR_BGR2RGB))

    if transformed_spots is not None:
        axes[2].scatter(
            transformed_spots[:, 0],
            transformed_spots[:, 1],
            s=10,
            color=COLOR_SOURCE,
            alpha=0.45,
        )

    for expected, estimate in zip(example.matched_target, predicted, strict=True):
        axes[2].plot(
            (expected[0], estimate[0]),
            (expected[1], estimate[1]),
            color="#ff5555",
            linewidth=0.8,
            alpha=0.8,
        )

    axes[2].scatter(
        example.matched_target[:, 0],
        example.matched_target[:, 1],
        s=35,
        color=COLOR_TARGET,
        label="Verified location",
    )
    axes[2].scatter(
        predicted[:, 0],
        predicted[:, 1],
        s=42,
        facecolors="none",
        edgecolors=COLOR_SOURCE,
        label="Prediction",
    )
    axes[2].legend(loc="upper right")
    axes[2].set_title("Held-out or label-free predictions and residuals")

    for axis in axes:
        axis.set_axis_off()

    figure.suptitle(title)
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


def show(matrix: np.ndarray, name: str) -> None:
    points = np.array(
        [[0, 0], [2, 0], [4, 0], [0, 2], [2, 2], [4, 2], [0, 4], [2, 4], [4, 4]],
        dtype=float,
    )
    connections = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8)]
    transformed = project_points(points, matrix)

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))

    for axis, values, title in zip(axes, (points, transformed), ("Original", name), strict=True):
        for indices in connections:
            axis.plot(values[list(indices), 0], values[list(indices), 1], "-o")

        for index, point in enumerate(values):
            axis.annotate(str(index), point + (0.12, 0.12))

        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-2, 10)
        axis.set_ylim(10, -2)
        axis.grid(alpha=0.2)
        axis.set_title(title)

    figure.tight_layout()
    plt.show()
