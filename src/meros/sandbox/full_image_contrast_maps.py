"""
Inspect signed, multiscale luminance contrast over entire subject photographs.

Run from the repository root:
    python -m meros.sandbox.full_image_contrast_maps

Blue means darker than the local neighborhood; red means brighter. These maps
are not spot segmentations. Nothing is saved to disk.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


IMAGES = (("rogerio", 9), ("rogerio", 12))
MAX_DISPLAY_WIDTH = 1400
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILES = (8, 8)
SCALES = ((2, 8), (5, 20), (12, 48))
DISPLAY_PERCENTILE = 97


def contrast_maps(image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    light = cv.cvtColor(image, cv.COLOR_BGR2LAB)[:, :, 0]
    enhanced = cv.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILES
    ).apply(light)
    values = enhanced.astype(np.float32)
    maps: list[np.ndarray] = []

    for inner_sigma, outer_sigma in SCALES:
        inner = cv.GaussianBlur(values, (0, 0), sigmaX=inner_sigma)
        outer = cv.GaussianBlur(values, (0, 0), sigmaX=outer_sigma)
        maps.append(inner - outer)

    return enhanced, maps


def show(image: np.ndarray, name: str) -> None:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / width)
    resized = cv.resize(image, (round(width * scale), round(height * scale)))
    enhanced, maps = contrast_maps(resized)

    figure, axes = plt.subplots(2, 3, figsize=(17, 11))

    axes[0, 0].imshow(cv.cvtColor(resized, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("CLAHE luminance")

    axes[0, 2].imshow(np.minimum(maps[0], 0), cmap="coolwarm", vmin=-max(
        1.0, float(np.percentile(np.abs(maps[0]), DISPLAY_PERCENTILE))
    ), vmax=max(1.0, float(np.percentile(np.abs(maps[0]), DISPLAY_PERCENTILE))))
    axes[0, 2].set_title("Fine scale: dark response only")

    for axis, response, (inner_sigma, outer_sigma) in zip(
        axes[1], maps, SCALES, strict=True
    ):
        limit = max(1.0, float(np.percentile(np.abs(response), DISPLAY_PERCENTILE)))
        plotted = axis.imshow(response, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"Contrast sigma {inner_sigma} - {outer_sigma}")
        figure.colorbar(plotted, ax=axis, fraction=0.045, pad=0.02)

    for axis in axes.flat:
        axis.set_axis_off()

    figure.suptitle(f"{name} - negative=blue, positive=red (per-panel scale)")
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


def run() -> None:
    project = Project.open()

    for subject_id, image_id in IMAGES:
        image = project.media.read_subject_image(subject_id, image_id)
        show(image, f"{subject_id}/{image_id}")


if __name__ == "__main__":
    run()
