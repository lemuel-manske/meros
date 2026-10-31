"""Demo: perspective distortion from a square to a quadrilateral."""

import cv2 as cv
import numpy as np

from meros.sandbox._geometry_demo import show


source = np.float32([[0, 0], [4, 0], [4, 4], [0, 4]])
target = np.float32([[1, 1], [6, 0], [5, 5], [0, 4]])

matrix = cv.getPerspectiveTransform(source, target)


if __name__ == "__main__":
    show(matrix, "Homography")
