"""Demo: perspective distortion from a square to a quadrilateral."""

import cv2 as cv
import numpy as np

from meros.sandbox.common import show


source = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float32)
target = np.array([[1, 1], [6, 0], [5, 5], [0, 4]], dtype=np.float32)

matrix = cv.getPerspectiveTransform(source, target)


if __name__ == "__main__":
    show(matrix, "Homography")
