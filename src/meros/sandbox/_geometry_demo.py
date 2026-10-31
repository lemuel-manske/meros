"""Small visual playground for 2D geometric transformations."""

import matplotlib.pyplot as plt
import numpy as np

from meros.ml.compare_spot_transforms import project_points


def show(matrix: np.ndarray, name: str) -> None:
    points = np.array(
        [[0, 0], [2, 0], [4, 0], [0, 2], [2, 2], [4, 2], [0, 4], [2, 4], [4, 4]],
        dtype=float,
    )
    connections = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8)]
    transformed = project_points(points, matrix)

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))

    for axis, values, title in zip(
        axes, (points, transformed), ("Original", name), strict=True
    ):
        for indices in connections:
            axis.plot(values[list(indices), 0], values[list(indices), 1], "-o")

        for index, point in enumerate(values):
            axis.annotate(str(index), point + (0.12, 0.12))

        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-2, 10)
        axis.set_ylim(10, -2)
        axis.grid(alpha=0.2)
        axis.set_title(title)

    figure.tight_layout()
    plt.show()


