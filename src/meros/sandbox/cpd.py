"""
Evaluate label-free non-rigid CPD on annotated grouper photographs.

CPD fits the complete, UNLABELED source and target spot sets. Eye and mouth
annotations establish an anatomical similarity initialization. Verified spot
pairs are read only after fitting, to assess geometric error and identity.
This benchmark is not comparable to supervised leave-one-out TPS as a ranking.

Run: python -m meros.sandbox.cpd
Requires DVC photos and the 10-12 spot-correspondence metadata.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist

from meros.project import Project
from meros.sandbox.spot_registration import load_example, show_predictions, summarize


SUBJECT = "rogerio"
SOURCE_ID = 10
TARGET_ID = 12

CPD_BETA_PX = 130.0
CPD_REGULARIZATION = 1.0
CPD_OUTLIER_FRACTION = 0.30
CPD_ITERATIONS = 150
CPD_MIN_VARIANCE = 1e-5
COORDINATE_SCALE = 100.0


def anatomical_similarity(
    source_eye: np.ndarray, source_mouth: np.ndarray,
    target_eye: np.ndarray, target_mouth: np.ndarray,
) -> np.ndarray:
    """
    Return a similarity matrix from two anatomical landmarks in each image.
    """
    source_axis = source_eye - source_mouth
    target_axis = target_eye - target_mouth
    denominator = float(source_axis @ source_axis)

    if denominator <= 0 or float(target_axis @ target_axis) <= 0:
        raise ValueError("Eye and mouth landmarks must be distinct")

    cosine = float(source_axis @ target_axis) / denominator
    sine = float(np.linalg.det(np.stack((source_axis, target_axis)))) / denominator
    linear = np.array(((cosine, -sine), (sine, cosine)))
    translation = target_eye - linear @ source_eye
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = translation
    return matrix


def nonrigid_cpd(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Fit standard Gaussian-kernel CPD, returning transformed unlabeled source spots.
    """
    y = source / COORDINATE_SCALE
    x = target / COORDINATE_SCALE
    beta = CPD_BETA_PX / COORDINATE_SCALE
    m, dimensions = y.shape
    n = len(x)
    kernel = np.exp(-cdist(y, y, "sqeuclidean") / (2 * beta**2))
    transformed = y.copy()
    variance = max(float(np.mean(cdist(x, y, "sqeuclidean")) / dimensions), 0.01)

    for iteration in range(CPD_ITERATIONS):
        distances = cdist(transformed, x, "sqeuclidean")
        probabilities = np.exp(-distances / (2 * variance))
        outlier_term = (
            (2 * np.pi * variance) ** (dimensions / 2)
            * CPD_OUTLIER_FRACTION / (1 - CPD_OUTLIER_FRACTION)
            * m / n
        )
        probabilities /= probabilities.sum(axis=0, keepdims=True) + outlier_term + 1e-80

        source_mass = probabilities.sum(axis=1)
        target_mass = probabilities.sum(axis=0)
        weighted_targets = probabilities @ x
        total_mass = float(source_mass.sum())

        if total_mass < 3:
            raise RuntimeError("CPD lost the soft correspondences")

        system = source_mass[:, None] * kernel + CPD_REGULARIZATION * variance * np.eye(m)
        weights = solve(system, weighted_targets - source_mass[:, None] * y, check_finite=False)
        transformed = y + kernel @ weights
        squared_error = (
            target_mass @ np.sum(x**2, axis=1)
            - 2 * np.sum(weighted_targets * transformed)
            + source_mass @ np.sum(transformed**2, axis=1)
        )
        updated_variance = max(float(squared_error / (total_mass * dimensions)), CPD_MIN_VARIANCE)

        if abs(updated_variance - variance) / variance < 1e-5:
            return transformed * COORDINATE_SCALE, iteration + 1

        variance = updated_variance

    return transformed * COORDINATE_SCALE, CPD_ITERATIONS


def run() -> None:
    project = Project.open()
    example = load_example(project, SUBJECT, SOURCE_ID, TARGET_ID)
    source_eye = project.metadata.read_subject_eye(SUBJECT, SOURCE_ID)
    target_eye = project.metadata.read_subject_eye(SUBJECT, TARGET_ID)
    source_mouth = project.metadata.read_subject_mouth(SUBJECT, SOURCE_ID)
    target_mouth = project.metadata.read_subject_mouth(SUBJECT, TARGET_ID)

    if source_eye is None or target_eye is None or source_mouth is None or target_mouth is None:
        raise ValueError("CPD requires eye and mouth annotations for both photographs")

    matrix = anatomical_similarity(
        np.asarray(source_eye.center), np.asarray(source_mouth),
        np.asarray(target_eye.center), np.asarray(target_mouth),
    )
    aligned = example.source_spots @ matrix[:2, :2].T + matrix[:2, 2]
    transformed, iterations = nonrigid_cpd(aligned, example.target_spots)

    # Associate verification labels to their nearest source annotation only AFTER fitting.
    distances = cdist(example.matched_source, example.source_spots)
    source_indices = distances.argmin(axis=1)
    deviations = distances.min(axis=1)
    predictions = transformed[source_indices]

    print(f"Unlabeled spots: {len(aligned)} -> {len(example.target_spots)}")
    print(f"CPD iterations: {iterations}; max label-to-source offset: {np.max(deviations):.1f}px")
    summarize("CPD (label-free)", predictions, example)
    show_predictions(
        example, predictions, "Non-rigid CPD: labels used only for evaluation",
        transformed_spots=transformed,
    )


if __name__ == "__main__":
    run()
