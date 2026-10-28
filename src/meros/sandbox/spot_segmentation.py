"""Visualize dark-spot segmentation and connected-component centroids."""

import cv2 as cv
import numpy as np

from meros.sandbox.spot_detection import display


# Edit these parameters and rerun this file to compare results.
BLOCK_SIZE = 35  # odd, greater than 1
THRESHOLD_OFFSET = 7
MIN_AREA = 8
MAX_AREA = 800
OPEN_KERNEL = 3


def detect(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    enhanced = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    binary = cv.adaptiveThreshold(
        enhanced,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        BLOCK_SIZE,
        THRESHOLD_OFFSET,
    )
    binary = cv.bitwise_and(binary, mask)
    kernel = np.ones((OPEN_KERNEL, OPEN_KERNEL), dtype=np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    binary = cv.bitwise_and(binary, mask)

    _, _, stats, centers = cv.connectedComponentsWithStats(binary)
    points = []

    for index in range(1, len(stats)):
        area = int(stats[index, cv.CC_STAT_AREA])

        if not MIN_AREA <= area <= MAX_AREA:
            continue

        x, y = centers[index]

        if mask[min(round(y), mask.shape[0] - 1), min(round(x), mask.shape[1] - 1)] != 0:
            points.append((float(x), float(y)))

    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


if __name__ == "__main__":
    display("Segmentation", detect)
