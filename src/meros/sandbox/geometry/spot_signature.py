"""
Coordinate-only spot signatures selected by the 2026 verification campaign.

These descriptors compare distributions, not point correspondences or fitted
geometric transformations. Scores are uncalibrated research evidence, not
probabilities of identity or automatic known/unknown decisions.

Historical E05: local three-neighbor distribution, energy distance.
Historical E24: the same local features with 0.25-weighted global context.

Run: python -m meros.sandbox.geometry.spot_signature
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from meros.project import Project


GLOBAL_WEIGHT = 0.25
SHOW_PLOTS = True
EXAMPLE_PAIRS = (
    (("rogerio", 10), ("rogerio", 12)),
    (("rafael", 3), ("rogerio", 9)),
)


def descriptors(points: np.ndarray, global_weight: float = GLOBAL_WEIGHT) -> np.ndarray:
    """
    Describe each spot using local neighborhood statistics and optional global context.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("Expected an (N, 2) array with at least four spots")

    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    indices = np.argsort(distances, axis=1)[:, :3]
    nearest = np.take_along_axis(distances, indices, axis=1)
    spacing = max(float(np.median(nearest[:, 0])), 1e-12)

    offsets = points[indices] - points[:, None, :]
    angles = np.arctan2(offsets[:, :, 1], offsets[:, :, 0])
    angular = np.column_stack(
        [np.abs(np.mean(np.exp(1j * order * angles), axis=1)) for order in (1, 2, 3)]
    )
    local = np.column_stack((np.log1p(nearest / spacing), angular))

    if global_weight == 0:
        return local

    np.fill_diagonal(distances, np.nan)
    centered = points - points.mean(axis=0)
    radius = max(float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))), 1e-12)
    global_context = np.nanquantile(distances, [0.25, 0.5, 0.75], axis=1).T / radius

    return np.column_stack((local, global_weight * global_context))


def score(
    source: np.ndarray,
    target: np.ndarray,
    global_weight: float = GLOBAL_WEIGHT,
) -> float:
    """
    Return an uncalibrated score; larger values indicate closer spot distributions.
    """
    first = descriptors(source, global_weight)
    second = descriptors(target, global_weight)

    within_first = cdist(first, first).mean()
    within_second = cdist(second, second).mean()
    between = cdist(first, second).mean()
    energy = max(0.0, float(2 * between - within_first - within_second))

    return -energy


def run() -> None:
    """
    Compare one known-positive and one known-negative pair for visual inspection.
    """
    project = Project.open()

    for source_id, target_id in EXAMPLE_PAIRS:
        source = np.asarray(project.metadata.read_subject_spots(*source_id), dtype=float)
        target = np.asarray(project.metadata.read_subject_spots(*target_id), dtype=float)

        local_score = score(source, target, global_weight=0.0)
        joint_score = score(source, target)
        print(
            f"{source_id} -> {target_id}: "
            f"E05.local3.energy={local_score:.6f}, "
            f"E24.global_joint0.25={joint_score:.6f}"
        )

        if not SHOW_PLOTS:
            continue

        figure, axes = plt.subplots(1, 2, figsize=(11, 5))

        for axis, points, label in zip(axes, (source, target), (source_id, target_id), strict=True):
            axis.scatter(points[:, 0], points[:, 1], s=16)
            axis.set_aspect("equal", adjustable="box")
            axis.invert_yaxis()
            axis.set_title(f"{label[0]}/{label[1]} ({len(points)} spots)")

        figure.suptitle(f"Coordinate-only comparison | E24 score: {joint_score:.4f}")
        figure.tight_layout()
        plt.show(block=True)
        plt.close(figure)


if __name__ == "__main__":
    run()
