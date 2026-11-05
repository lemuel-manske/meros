"""
Inspect faint head-marking constellations at multiple spatial scales.

Run from the repository root:
    python -m meros.sandbox.local_head_constellation

Select a rectangular region in each photograph (drag, then press Enter or Space).
Press Escape or cancel to skip a photograph. Selected regions are hypotheses
for inspection, not verified correspondences. Nothing is saved to disk or used
to predict individual identity.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import maximum_filter

from meros.project import Project


SUBJECT = "rogerio"
IMAGE_IDS = (9, 12)
MAX_SELECTION_SIDE = 1200

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (5, 5)

FINE_SIGMA_FRACTION = 0.018
MEDIUM_SIGMA_FRACTION = 0.075
BACKGROUND_SIGMA_FRACTION = 0.22

PEAK_SPACING_FRACTION = 0.12
PEAK_BORDER_FRACTION = 0.08
PEAK_MIN_PERCENTILE = 80
MAX_DISPLAY_PEAKS = 12


def select_region(image: np.ndarray, image_id: int) -> np.ndarray | None:
    """
    Select a region interactively on a scaled display of the original photograph.
    """
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SELECTION_SIDE / max(height, width))
    shown = cv.resize(image, None, fx=scale, fy=scale) if scale < 1 else image

    window = f"{SUBJECT}/{image_id}: select region (Enter/Space confirms, Esc skips)"
    x, y, w, h = cv.selectROI(window, shown, showCrosshair=True, fromCenter=False)
    cv.destroyWindow(window)

    if w <= 0 or h <= 0:
        return None

    x1 = max(0, round(x / scale))
    y1 = max(0, round(y / scale))
    x2 = min(width, round((x + w) / scale))
    y2 = min(height, round((y + h) / scale))

    return image[y1:y2, x1:x2]


def responses(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enhance local contrast and retain positive/negative deviations at two scales.
    """
    light = cv.cvtColor(image, cv.COLOR_BGR2LAB)[:, :, 0]
    equalized = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID).apply(
        light
    )

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
    selections: dict[int, np.ndarray] = {}

    for image_id in IMAGE_IDS:
        image = project.media.read_subject_image(SUBJECT, image_id)
        crop = select_region(image, image_id)

        if crop is not None:
            selections[image_id] = crop

    if not selections:
        print("No regions selected.")
        return

    figure, axes = plt.subplots(
        5, len(selections), figsize=(6 * len(selections), 17), squeeze=False
    )

    for column, (image_id, crop) in enumerate(selections.items()):
        equalized, fine, medium = responses(crop)
        peaks = candidate_peaks(fine)

        print(f"{SUBJECT}/{image_id}: {len(peaks)} positive local maxima displayed")

        axis = axes[0, column]
        axis.imshow(cv.cvtColor(crop, cv.COLOR_BGR2RGB))
        axis.set_title(f"{SUBJECT}/{image_id}: selected region")

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
