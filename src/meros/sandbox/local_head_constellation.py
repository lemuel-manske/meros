"""
Inspect faint head-marking constellations at multiple spatial scales.

Run from the repository root:
    python -m meros.sandbox.local_head_constellation

The two regions were located visually in the original Rogerio/9 and Rogerio/12
photographs. They are hypotheses for inspection, not verified correspondences.
Nothing is saved to disk or used to predict individual identity.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import maximum_filter

from meros.project import Project


SUBJECT = "rogerio"
REGIONS = {
    9: (510, 333, 622, 424),
    12: (744, 530, 997, 737),
}

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (5, 5)

FINE_SIGMA_FRACTION = 0.018
MEDIUM_SIGMA_FRACTION = 0.075
BACKGROUND_SIGMA_FRACTION = 0.22

PEAK_SPACING_FRACTION = 0.12
PEAK_BORDER_FRACTION = 0.08
PEAK_MIN_PERCENTILE = 80
MAX_DISPLAY_PEAKS = 12


def responses(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhance local contrast and retain positive/negative deviations at two scales.
    """
    light = cv.cvtColor(image, cv.COLOR_BGR2LAB)[:, :, 0]
    equalized = cv.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
    ).apply(light)

    side = min(equalized.shape)
    values = equalized.astype(np.float32)

    fine_sigma = max(1, round(side * FINE_SIGMA_FRACTION))
    medium_sigma = max(2, round(side * MEDIUM_SIGMA_FRACTION))
    background_sigma = max(4, round(side * BACKGROUND_SIGMA_FRACTION))

    fine = cv.GaussianBlur(values, (0, 0), sigmaX=fine_sigma)
    medium = cv.GaussianBlur(values, (0, 0), sigmaX=medium_sigma)
    background = cv.GaussianBlur(values, (0, 0), sigmaX=background_sigma)

    return equalized, fine - background, medium - background


def candidate_peaks(response: np.ndarray) -> np.ndarray:
    """
    Find positive local maxima for visual inspection; do not force three peaks.
    """
    side = min(response.shape)
    spacing = max(3, round(side * PEAK_SPACING_FRACTION))
    border = max(2, round(side * PEAK_BORDER_FRACTION))

    neighborhood = maximum_filter(response, size=spacing * 2 + 1, mode="nearest")
    cutoff = max(0.0, float(np.percentile(response, PEAK_MIN_PERCENTILE)))

    candidates = (response == neighborhood) & (response > cutoff)
    candidates[:border] = False
    candidates[-border:] = False
    candidates[:, :border] = False
    candidates[:, -border:] = False

    ys, xs = np.nonzero(candidates)
    order = np.argsort(-response[ys, xs])[:MAX_DISPLAY_PEAKS]

    return np.column_stack((xs[order], ys[order]))


def run() -> None:
    project = Project.open()
    figure, axes = plt.subplots(5, len(REGIONS), figsize=(12, 17))

    for column, (image_id, (x1, y1, x2, y2)) in enumerate(REGIONS.items()):
        image = project.media.read_subject_image(SUBJECT, image_id)
        height, width = image.shape[:2]

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid local region for {SUBJECT}/{image_id}")

        crop = image[y1:y2, x1:x2]
        equalized, fine, medium = responses(crop)
        peaks = candidate_peaks(fine)

        print(f"{SUBJECT}/{image_id}: {len(peaks)} positive local maxima displayed")

        axis = axes[0, column]
        axis.imshow(cv.cvtColor(crop, cv.COLOR_BGR2RGB))
        axis.set_title(f"{SUBJECT}/{image_id}: original")

        axis = axes[1, column]
        axis.imshow(equalized, cmap="gray", vmin=0, vmax=255)
        axis.set_title("CLAHE luminance")

        for row, (response, title) in enumerate(
            ((fine, "Fine-scale contrast"), (medium, "Medium-scale contrast")), start=2
        ):
            limit = max(1.0, float(np.percentile(np.abs(response), 95)))
            axis = axes[row, column]
            axis.imshow(response, cmap="coolwarm", vmin=-limit, vmax=limit)
            axis.set_title(title)

        axis = axes[4, column]
        axis.imshow(equalized, cmap="gray", vmin=0, vmax=255)

        if len(peaks):
            axis.scatter(
                peaks[:, 0],
                peaks[:, 1],
                s=46,
                facecolors="none",
                edgecolors="#ffff00",
                linewidths=1.3,
            )

        axis.set_title("Candidate positive maxima (not matched)")

    for axis in axes.flat:
        axis.set_axis_off()

    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


if __name__ == "__main__":
    run()
