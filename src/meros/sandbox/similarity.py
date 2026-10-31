"""Demo: rotation, uniform scaling and translation."""

from math import cos, radians, sin

import numpy as np

from meros.sandbox._geometry_demo import show


ANGLE = radians(25)
SCALE = 1.2
TX, TY = 1.5, 0.8

matrix = np.array(
    [
        [SCALE * cos(ANGLE), -SCALE * sin(ANGLE), TX],
        [SCALE * sin(ANGLE), SCALE * cos(ANGLE), TY],
        [0, 0, 1],
    ],
    dtype=float,
)


if __name__ == "__main__":
    show(matrix, "Similarity")
