"""
Explore CLAHE-enhanced head-pattern structures without manual spot annotations.

Run from the repository root after restoring project media:
    python -m meros.sandbox.head_pattern_features

Displays the CLAHE head crops, a medium-scale contrast map, and automatically
detected SIFT regions. No matching, identity scoring, or output files.
"""

import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from meros.project import Project


SUBJECT = "rogerio"

CROPS = {
    9: (270, 118, 962, 616),
    12: (165, 115, 1680, 1120),
}

WIDTH = 900
HEIGHT = 630

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (9, 7)

DETECTOR_BLUR_SIGMA = 4.0
MIN_FEATURE_SIZE = 24.0
MIN_FEATURE_SPACING = 26.0
MAX_FEATURES = 40

RESPONSE_SMALL_SIGMA = 5.0
RESPONSE_LARGE_SIGMA = 28.0


def fit_canvas(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]

    scale = min(WIDTH / width, HEIGHT / height)
    interpolation = cv.INTER_AREA if scale < 1 else cv.INTER_CUBIC

    scaled = cv.resize(
        image, (round(width * scale), round(height * scale)), interpolation=interpolation
    )

    canvas = np.full((HEIGHT, WIDTH, 3), (28, 28, 28), dtype=np.uint8)

    y = (HEIGHT - scaled.shape[0]) // 2
    x = (WIDTH - scaled.shape[1]) // 2

    canvas[y : y + scaled.shape[0], x : x + scaled.shape[1]] = scaled

    return canvas


def enhance(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    light, channel_a, channel_b = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))

    luminance = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID).apply(
        light
    )

    color = cv.cvtColor(cv.merge((luminance, channel_a, channel_b)), cv.COLOR_LAB2BGR)

    return color, luminance


def contrast_map(luminance: np.ndarray) -> np.ndarray:
    """
    Visualize medium-scale tonal changes; not a segmentation mask.
    """
    values = luminance.astype(np.float32)

    small = cv.GaussianBlur(values, (0, 0), sigmaX=RESPONSE_SMALL_SIGMA)
    large = cv.GaussianBlur(values, (0, 0), sigmaX=RESPONSE_LARGE_SIGMA)
    difference = small - large

    lower, upper = np.percentile(difference, (3, 97))

    return np.clip((difference - lower) * 255 / max(upper - lower, 1e-6), 0, 255).astype(np.uint8)


def detect_regions(luminance: np.ndarray) -> list[cv.KeyPoint]:
    """
    Detect larger SIFT regions on smoothed CLAHE luminance.
    """
    smoothed = cv.GaussianBlur(luminance, (0, 0), sigmaX=DETECTOR_BLUR_SIGMA)

    detector = getattr(cv, "SIFT_create")(nfeatures=450, contrastThreshold=0.015)
    candidates = detector.detect(smoothed, None)

    candidates = [
        point
        for point in candidates
        if point.size >= MIN_FEATURE_SIZE
        and 20 <= point.pt[0] <= WIDTH - 20
        and 20 <= point.pt[1] <= HEIGHT - 20
    ]
    candidates.sort(key=lambda point: point.response, reverse=True)

    regions: list[cv.KeyPoint] = []

    for point in candidates:
        if any(
            (point.pt[0] - other.pt[0]) ** 2 + (point.pt[1] - other.pt[1]) ** 2
            < MIN_FEATURE_SPACING**2
            for other in regions
        ):
            continue

        regions.append(point)

        if len(regions) >= MAX_FEATURES:
            break

    return regions


def run() -> None:
    project = Project.open()
    figure, axes = plt.subplots(3, len(CROPS), figsize=(15, 13))

    for column, (image_id, (x1, y1, x2, y2)) in enumerate(CROPS.items()):
        image = project.media.read_subject_image(SUBJECT, image_id)

        height, width = image.shape[:2]

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Crop for {SUBJECT}/{image_id} exceeds source image bounds")

        crop = fit_canvas(image[y1:y2, x1:x2])
        enhanced, luminance = enhance(crop)
        response = contrast_map(luminance)
        regions = detect_regions(luminance)

        print(f"{SUBJECT}/{image_id}: {len(regions)} detected regions")

        rgb = cv.cvtColor(enhanced, cv.COLOR_BGR2RGB)

        axes[0, column].imshow(rgb)
        axes[0, column].set_title(f"{SUBJECT}/{image_id} — CLAHE")

        axes[1, column].imshow(response, cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title("Medium-scale contrast (5–28 px)")

        axes[2, column].imshow(rgb)

        for point in regions:
            circle = Circle(
                (float(point.pt[0]), float(point.pt[1])),
                radius=point.size / 2,
                fill=False,
                edgecolor="#ffdd44",
                linewidth=1,
                alpha=0.8,
            )
            axes[2, column].add_patch(circle)

        axes[2, column].set_title(f"Automatically detected regions: {len(regions)}")

    for axis in axes.flat:
        axis.set_axis_off()

    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


if __name__ == "__main__":
    run()
