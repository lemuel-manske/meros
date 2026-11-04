"""
Compare global transforms and regularized TPS on manually matched grouper spots.

Each evaluated correspondence is held out of fitting. TPS smoothing is chosen
using only the remaining training pairs (nested cross-validation). This is a
supervised geometric benchmark, not automatic fish identification.

Run: python -m meros.sandbox.geometry.tps
Requires DVC photos and the 10-12 spot-correspondence metadata.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from scipy.interpolate import RBFInterpolator

from meros.project import Project
from meros.sandbox.geometry.common import load_example, show_predictions, summarize


SUBJECT = "rogerio"
SOURCE_ID = 10
TARGET_ID = 12

SMOOTHING_VALUES = (0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0)
INNER_FOLDS = 4
VALIDATION = "leave_one_out"  # "leave_one_out" or "spatial"
SPATIAL_BLOCKS = 5
SEED = 17


def fit_transform(source: np.ndarray, target: np.ndarray, method: str, smoothing: float = 0.0):
    """
    Fit a supervised 2D transform and return a function projecting new points.
    """
    if method == "Similarity":
        count = len(source)
        design = np.zeros((2 * count, 4))
        design[0::2, :] = np.column_stack(
            (source[:, 0], -source[:, 1], np.ones(count), np.zeros(count))
        )
        design[1::2, :] = np.column_stack(
            (source[:, 1], source[:, 0], np.zeros(count), np.ones(count))
        )
        a, b, tx, ty = np.linalg.lstsq(design, target.ravel(), rcond=None)[0]

        def similarity(points: np.ndarray) -> np.ndarray:
            x, y = points[:, 0], points[:, 1]
            return np.column_stack((a * x - b * y + tx, b * x + a * y + ty))

        return similarity

    if method == "Affine":
        design = np.column_stack((source, np.ones(len(source))))
        matrix = np.linalg.lstsq(design, target, rcond=None)[0]
        return lambda points: np.column_stack((points, np.ones(len(points)))) @ matrix

    if method == "Homography":
        matrix, _ = cv.findHomography(source, target, method=0)

        if matrix is None:
            raise ValueError("Could not estimate homography")

        return lambda points: cv.perspectiveTransform(
            np.asarray(points, dtype=np.float64).reshape(1, -1, 2), matrix
        ).reshape(-1, 2)

    if method == "Quadratic":
        center = source.mean(axis=0)
        scale = max(float(np.sqrt(np.mean(np.sum((source - center) ** 2, axis=1)))), 1e-9)

        def features(points: np.ndarray) -> np.ndarray:
            x, y = ((points - center) / scale).T
            return np.column_stack((np.ones(len(points)), x, y, x * x, x * y, y * y))

        coefficients = np.linalg.lstsq(features(source), target, rcond=None)[0]
        return lambda points: features(points) @ coefficients

    if method == "TPS":
        center = source.mean(axis=0)
        scale = max(float(np.sqrt(np.mean(np.sum((source - center) ** 2, axis=1)))), 1e-9)
        interpolator = RBFInterpolator(
            (source - center) / scale,
            target,
            kernel="thin_plate_spline",
            degree=1,
            smoothing=smoothing,
        )
        return lambda points: interpolator((points - center) / scale)

    raise ValueError(f"Unknown transform: {method}")


def choose_smoothing(source: np.ndarray, target: np.ndarray) -> float:
    """
    Select TPS regularization by four-fold CV strictly within the training set.
    """
    order = np.random.default_rng(SEED).permutation(len(source))
    folds = np.array_split(order, INNER_FOLDS)
    scores = []

    for smoothing in SMOOTHING_VALUES:
        fold_errors = []

        for held_out in folds:
            training = np.ones(len(source), dtype=bool)
            training[held_out] = False

            try:
                predict = fit_transform(source[training], target[training], "TPS", smoothing)
                residual = np.linalg.norm(predict(source[held_out]) - target[held_out], axis=1)
            except np.linalg.LinAlgError:
                residual = np.full(len(held_out), 1e9)

            fold_errors.extend(residual)

        scores.append(float(np.median(fold_errors) + 0.15 * np.mean(fold_errors)))

    return SMOOTHING_VALUES[int(np.argmin(scores))]


def validation_groups(source: np.ndarray) -> np.ndarray:
    """
    Reserve individual matches or contiguous horizontal regions for validation.
    """
    if VALIDATION == "leave_one_out":
        return np.arange(len(source))

    if VALIDATION == "spatial":
        borders = np.quantile(source[:, 0], np.linspace(0, 1, SPATIAL_BLOCKS + 1)[1:-1])
        return np.searchsorted(borders, source[:, 0])

    raise ValueError(f"Unknown validation: {VALIDATION}")


def cross_validate(source: np.ndarray, target: np.ndarray, method: str) -> np.ndarray:
    """
    Predict only on pairs omitted from the corresponding model fit.
    """
    predictions = np.empty_like(target)
    groups = validation_groups(source)

    for group in np.unique(groups):
        held_out = groups == group
        training = ~held_out
        smoothing = choose_smoothing(source[training], target[training]) if method == "TPS" else 0.0
        predict = fit_transform(source[training], target[training], method, smoothing)
        predictions[held_out] = predict(source[held_out])

    return predictions


def run() -> None:
    example = load_example(Project.open(), SUBJECT, SOURCE_ID, TARGET_ID)
    source, target = example.matched_source, example.matched_target

    print(f"Verified pairs: {len(source)} | validation: {VALIDATION}")

    for method in ("Similarity", "Affine", "Homography", "Quadratic", "TPS"):
        predictions = cross_validate(source, target, method)
        summarize(method, predictions, example)

        if method == "TPS":
            show_predictions(example, predictions, f"TPS: {VALIDATION} (unseen pairs)")


if __name__ == "__main__":
    run()
