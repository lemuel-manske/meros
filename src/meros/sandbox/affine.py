"""Demo: independent scaling, shear and translation."""

import numpy as np

from meros.sandbox._geometry_demo import show


matrix = np.array(
    [
        [1.3, 0.45, 1.0],
        [0.15, 0.85, 0.5],
        [0, 0, 1],
    ],
    dtype=float,
)


if __name__ == "__main__":
    show(matrix, "Affine")
