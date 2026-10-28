"""Experiment with scale-normalized LoG and local contrast normalization.

Standalone sandbox prototype: does not change the original spot_log baseline.
"""

import cv2 as cv
import numpy as np

from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter

from meros.sandbox.spot_detection import display


# Parameters are intentionally edited here, not exposed as CLI arguments.
SIGMAS = (1.5, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 15.0)
PREPROCESS = "local"  # "local", "clahe", "none"
CLAHE_CLIP = 2.0
LOCAL_BACKGROUND_SIGMA = 15.0
LOCAL_CONTRAST_SIGMA = 8.0
MIN_RESPONSE = 1.2  # Response in locally normalized units when PREPROCESS="local".
MIN_RELATIVE_CONTRAST = 0.10
NMS_RADIUS_FACTOR = 0.8
MIN_SPACING_PX = 3.0
MAX_CANDIDATES = 6000  # Early cap after score sorting, for interactive speed.


def normalized_image(gray: np.ndarray) -> np.ndarray:
    image = gray.astype(np.float32) / 255.0

    if PREPROCESS == "none":
        return image

    if PREPROCESS == "clahe":
        enhanced = cv.createCLAHE(
            clipLimit=CLAHE_CLIP, tileGridSize=(8, 8)
        ).apply(gray)

        return enhanced.astype(np.float32) / 255.0

    if PREPROCESS != "local":
        raise ValueError(f"Unknown preprocessing: {PREPROCESS}")

    background = gaussian_filter(image, LOCAL_BACKGROUND_SIGMA)
    residual = image - background
    local_variance = gaussian_filter(residual * residual, LOCAL_CONTRAST_SIGMA)
    local_std = np.sqrt(np.maximum(local_variance, 1e-5))

    return residual / local_std


def detect_with_details(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    min_response: float = MIN_RESPONSE,
    min_relative_contrast: float = MIN_RELATIVE_CONTRAST,
    nms_radius_factor: float = NMS_RADIUS_FACTOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, scale estimates, and saliency scores in descending order."""
    if min_response < 0 or min_relative_contrast < 0 or nms_radius_factor <= 0:
        raise ValueError("Response and contrast must be nonnegative; NMS factor positive")

    enhanced = normalized_image(gray)
    responses = np.stack([
        sigma**2 * gaussian_laplace(enhanced, sigma=sigma)
        for sigma in SIGMAS
    ])

    valid = mask != 0
    peaks = responses == maximum_filter(responses, size=(3, 3, 3), mode="nearest")
    response_threshold = min_response if PREPROCESS == "local" else min_response / 255.0
    peaks &= responses >= response_threshold
    peaks &= valid[None, :, :]

    scales, ys, xs = np.nonzero(peaks)
    if not len(xs):
        empty = np.empty((0,), dtype=np.float32)

        return np.empty((0, 2), dtype=np.float32), empty, empty

    response_values = responses[scales, ys, xs]
    sigma_values = np.asarray(SIGMAS, dtype=np.float32)[scales]
    # A dark blob should be locally darker than its broader neighborhood.
    local_mean = gaussian_filter(enhanced, 2.0 * max(SIGMAS))
    contrast = local_mean[ys, xs] - enhanced[ys, xs]
    contrast_threshold = (
        min_relative_contrast if PREPROCESS == "local" else min_relative_contrast / 255.0
    )
    accepted = contrast >= contrast_threshold
    scores = response_values[accepted]
    sigma_values = sigma_values[accepted]
    positions = np.column_stack((xs[accepted], ys[accepted])).astype(np.float32)
    ranking = np.argsort(scores)[::-1][:MAX_CANDIDATES]

    selected: list[int] = []
    # Spatial hash grid bounds work compared with pairwise quadratic suppression.
    cell_size = max(1.0, nms_radius_factor * max(SIGMAS))
    cells: dict[tuple[int, int], list[int]] = {}

    for index in ranking:
        x, y = positions[index]
        radius = max(MIN_SPACING_PX, nms_radius_factor * sigma_values[index])
        cell_x, cell_y = int(x // cell_size), int(y // cell_size)
        reach = int(np.ceil((radius + nms_radius_factor * max(SIGMAS)) / cell_size))
        duplicate = False

        for ny in range(cell_y - reach, cell_y + reach + 1):
            if duplicate:
                break

            for nx in range(cell_x - reach, cell_x + reach + 1):
                for previous in cells.get((nx, ny), []):
                    distance_squared = float(np.sum((positions[index] - positions[previous]) ** 2))
                    separation = max(
                        radius,
                        MIN_SPACING_PX,
                        nms_radius_factor * sigma_values[previous],
                    )

                    if distance_squared < separation**2:
                        duplicate = True
                        break

                if duplicate:
                    break

        if duplicate:
            continue

        selected.append(int(index))
        cells.setdefault((cell_x, cell_y), []).append(int(index))

    selected_indices = np.asarray(selected, dtype=int)

    return (
        positions[selected_indices],
        sigma_values[selected_indices],
        scores[selected_indices].astype(np.float32),
    )


def detect(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    points, _, _ = detect_with_details(gray, mask)

    return points


if __name__ == "__main__":
    display("Adaptive LoG", detect)
