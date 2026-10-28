"""Visualize multi-scale Laplacian-of-Gaussian dark-spot centers."""

import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter

from meros.sandbox.spot_detection import display


# Edit these parameters and rerun this file to compare results.
SIGMAS = (2.0, 3.0, 4.5, 6.0, 8.0, 11.0)
MIN_RESPONSE = 12.0
MIN_SPACING_PX = 6


def detect(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    responses = np.stack(
        [
            -(sigma**2) * gaussian_laplace(normalized.astype(np.float32), sigma=sigma)
            for sigma in SIGMAS
        ]
    )
    peaks = responses == maximum_filter(responses, size=(3, 5, 5), mode="nearest")
    peaks &= responses >= MIN_RESPONSE
    peaks &= (mask != 0)[None, :, :]

    scales, ys, xs = np.nonzero(peaks)
    order = np.argsort(responses[scales, ys, xs])[::-1]
    selected: list[tuple[float, float]] = []

    for index in order:
        x = float(xs[index])
        y = float(ys[index])

        if all((x - px) ** 2 + (y - py) ** 2 >= MIN_SPACING_PX**2 for px, py in selected):
            selected.append((x, y))

    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


if __name__ == "__main__":
    display("LoG", detect)
