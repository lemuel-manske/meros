"""
Morphology-guided SIFT extraction for dark spots on a grouper head.

This module implements heuristic candidate selection; scores are not calibrated
probabilities and therefore cannot guarantee 90% precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class SpotSiftResult:
    spots: np.ndarray
    keypoints: tuple[cv.KeyPoint, ...]
    descriptors: np.ndarray | None
    spot_mask: np.ndarray


def extract_spot_sift(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    min_area: int = 9,
    max_area_fraction: float = 0.001,
    min_circularity: float = 0.55,
    min_contrast: float = 8.0,
    region_radius: int = 12,
    max_features: int = 4000,
    max_descriptors: int = 2500,
    mode: str = "flexible",
    scale_tolerance: float = 1.5,
) -> SpotSiftResult:
    """
    Detect plausible dark spots, then restrict SIFT features to their centers.

    Pass a head-only mask to exclude other anatomical regions. The returned
    descriptors correspond one-to-one with the filtered keypoints. In flexible
    mode, SIFT runs over the whole valid mask; spot-associated features are
    ranked first, and the remainder may fill the descriptor budget.
    """
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("Expected a uint8 grayscale, BGR or BGRA image")

    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Expected a uint8 grayscale, BGR or BGRA image")

    if min_area < 1 or not 0 < max_area_fraction <= 1:
        raise ValueError("Invalid spot area limits")

    if not 0 <= min_circularity <= 1 or min_contrast < 0 or region_radius < 1:
        raise ValueError("Invalid spot morphology parameters")

    if max_features < 1 or max_descriptors < 1 or scale_tolerance <= 0:
        raise ValueError("Invalid SIFT extraction parameters")

    if mode not in ("strict", "flexible"):
        raise ValueError("mode must be strict or flexible")

    gray = (
        image
        if image.ndim == 2
        else cv.cvtColor(image, cv.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv.COLOR_BGR2GRAY)
    )

    height, width = gray.shape
    valid = np.full((height, width), 255, dtype=np.uint8)

    if image.ndim == 3 and image.shape[2] == 4:
        valid = cv.bitwise_and(valid, image[:, :, 3])

    if mask is not None:
        if mask.shape != gray.shape:
            raise ValueError("Mask must have the same height and width as the image")

        valid = cv.bitwise_and(valid, (mask > 0).astype(np.uint8) * 255)

    valid = cv.erode(valid, np.ones((5, 5), dtype=np.uint8))

    if not np.any(valid):
        return SpotSiftResult(np.empty((0, 2), np.float32), (), None, np.zeros_like(gray))

    filled = gray.copy()
    filled[valid == 0] = int(np.median(gray[valid != 0]))
    enhanced = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(filled)
    binary = cv.adaptiveThreshold(
        enhanced, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 35, 8
    )
    binary[valid == 0] = 0

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    spots: list[tuple[float, float]] = []
    max_area = max(min_area, int(width * height * max_area_fraction))

    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)

        if not min_area <= area <= max_area or perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < min_circularity:
            continue

        moments = cv.moments(contour)

        if moments["m00"] == 0:
            continue

        x = float(moments["m10"] / moments["m00"])
        y = float(moments["m01"] / moments["m00"])
        center = (round(x), round(y))

        if not 0 <= center[0] < width or not 0 <= center[1] < height:
            continue

        if valid[center[1], center[0]] == 0:
            continue

        interior = np.zeros_like(gray)
        cv.drawContours(interior, [contour], -1, 255, -1)
        outer = cv.dilate(interior, np.ones((13, 13), dtype=np.uint8))
        ring = cv.bitwise_and(cv.subtract(outer, interior), valid)
        inside = (interior != 0) & (valid != 0)

        if not np.any(inside) or not np.any(ring):
            continue

        contrast = float(np.median(gray[ring != 0]) - np.median(gray[inside]))

        if contrast >= min_contrast:
            spots.append((x, y))

    spots.sort(key=lambda item: (item[1], item[0]))
    coordinates = np.asarray(spots, dtype=np.float32).reshape(-1, 2)
    spot_mask = np.zeros_like(gray)

    for x, y in coordinates:
        cv.circle(spot_mask, (round(float(x)), round(float(y))), region_radius, 255, -1)

    spot_mask = cv.bitwise_and(spot_mask, valid)

    if mode == "strict" and len(coordinates) == 0:
        return SpotSiftResult(coordinates, (), None, spot_mask)

    sift = cv.SIFT.create(nfeatures=max_features, contrastThreshold=0.025)
    detection_mask = spot_mask if mode == "strict" else valid
    keypoints, descriptors = sift.detectAndCompute(gray, detection_mask)

    if mode == "flexible":
        # Prioritize features associated with spots without throwing away
        # other descriptors that might remain repeatable in poor imagery.
        def priority(point: cv.KeyPoint) -> tuple[bool, float]:
            if len(coordinates) == 0:
                return (True, -point.response)

            nearest = float(np.min(np.linalg.norm(coordinates - point.pt, axis=1)))
            tolerance = max(region_radius, scale_tolerance * point.size)
            return (nearest > tolerance, -point.response)

        keep = sorted(range(len(keypoints)), key=lambda index: priority(keypoints[index]))
    else:
        keep = [
            index
            for index, point in enumerate(keypoints)
            if np.any(np.sum((coordinates - point.pt) ** 2, axis=1) <= region_radius**2)
        ]

    keep = keep[:max_descriptors]

    filtered = tuple(keypoints[index] for index in keep)
    selected = descriptors[keep] if descriptors is not None and keep else None

    return SpotSiftResult(coordinates, filtered, selected, spot_mask)
