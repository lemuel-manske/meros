"""Visualize OpenCV SimpleBlobDetector centers on masked frontal crops."""

import cv2 as cv
import numpy as np

from meros.sandbox.spot_detection import display


# Edit these parameters and rerun this file to compare results.
MIN_AREA = 8.0
MAX_AREA = 800.0
MIN_CIRCULARITY = 0.3
MIN_REPEATABILITY = 2


def detect(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    params = cv.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = MIN_AREA
    params.maxArea = MAX_AREA
    params.filterByCircularity = True
    params.minCircularity = MIN_CIRCULARITY
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minRepeatability = MIN_REPEATABILITY

    detector = cv.SimpleBlobDetector.create(params)
    enhanced = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    points = detector.detect(enhanced, mask)

    return np.asarray([point.pt for point in points], dtype=np.float32).reshape(-1, 2)


if __name__ == "__main__":
    display("SimpleBlobDetector", detect)
