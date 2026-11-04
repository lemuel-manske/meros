from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


# TODO: consider using a threshold based on the image scale
# currently, the RANSAC threshold is set to 12px, PIXELS
# in the future, we should consider using a threshold based on the image scale
RANSAC_THRESHOLD_PX = 12.0

SHOW_PLOTS = True


@dataclass(frozen=True)
class TransformModel:
    name: str
    minimum_points: int


MODELS = (
    TransformModel("similarity", 2),
    TransformModel("affine", 3),
    TransformModel("homography", 4),
)


def estimate(
    source: np.ndarray,
    target: np.ndarray,
    model: TransformModel,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(source) < model.minimum_points:
        return None

    try:
        if model.name == "similarity":
            matrix, mask = cv.estimateAffinePartial2D(
                source, target, method=cv.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PX
            )
        elif model.name == "affine":
            matrix, mask = cv.estimateAffine2D(
                source, target, method=cv.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD_PX
            )
        else:
            matrix, mask = cv.findHomography(source, target, cv.RANSAC, RANSAC_THRESHOLD_PX)

        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return None

        if model.name != "homography":
            matrix = np.vstack((matrix, [0.0, 0.0, 1.0]))

        return matrix, mask.ravel().astype(bool)
    except cv.error:
        return None


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = cv.perspectiveTransform(
            np.asarray(points, dtype=np.float64).reshape(1, -1, 2), matrix
        ).reshape(-1, 2)

    return projected
