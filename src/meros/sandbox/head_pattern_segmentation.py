"""
Experiment with a binary mask of pale skin markings on Rogerio/9 and Rogerio/12.

Run from the repository root after setting up SAM2:
    python -m meros.sandbox.head_pattern_segmentation

SAM2 limits the region to the fish, not to skin alone. The binary mask is
an exploratory candidate pattern, not verified biological segmentation.
Nothing is saved to disk.
"""

from importlib import import_module
from typing import Any

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


SUBJECT = "rogerio"
HEAD_BOXES = {
    9: (270, 118, 962, 616),
    12: (165, 115, 1680, 1120),
}

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (9, 7)
PATTERN_SIGMA = 4.0
ILLUMINATION_SIGMA = 35.0
MASK_EDGE_MARGIN = 12
MORPHOLOGY_SIZE = 7
MIN_REGION_FRACTION = 0.0003
MAX_REGION_FRACTION = 0.08


def image_predictor(checkpoint: str) -> Any:
    torch = import_module("torch")
    builder = import_module("sam2.build_sam")
    module = import_module("sam2.sam2_image_predictor")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = builder.build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path=checkpoint,
        device=device,
    )

    return module.SAM2ImagePredictor(model)


def subject_mask(predictor: Any, image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    masks, _, _ = predictor.predict(
        box=np.asarray(box, dtype=np.float32),
        multimask_output=False,
    )

    return np.asarray(masks[0], dtype=bool)


def enhance(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    light, channel_a, channel_b = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))
    clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    equalized = clahe.apply(light)
    color = cv.cvtColor(cv.merge((equalized, channel_a, channel_b)), cv.COLOR_LAB2BGR)

    return color, equalized


def binary_pattern(luminance: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Segment light, medium-scale skin markings as a single binary foreground.

    Contrast is measured relative to neighboring skin. The Otsu threshold
    is capped by a relaxed percentile to retain pale, low-contrast regions.
    Normalized convolution avoids bias from the area outside the SAM2 mask.
    """
    margin = 2 * MASK_EDGE_MARGIN + 1
    edge_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (margin, margin))
    interior = cv.erode(valid.astype(np.uint8) * 255, edge_kernel) > 0

    if not np.any(interior):
        raise ValueError("SAM2 mask has no interior area for pattern segmentation")

    values = luminance.astype(np.float32)
    weights = valid.astype(np.float32)

    weighted_light = cv.GaussianBlur(values * weights, (0, 0), sigmaX=ILLUMINATION_SIGMA)
    weighted_area = cv.GaussianBlur(weights, (0, 0), sigmaX=ILLUMINATION_SIGMA)
    background = weighted_light / np.maximum(weighted_area, 1e-6)

    smoothed = cv.GaussianBlur(values, (0, 0), sigmaX=PATTERN_SIGMA)
    residual = smoothed - background

    low, high = np.percentile(residual[interior], (2, 98))
    scaled = np.clip((residual - low) * 255 / max(high - low, 1e-6), 0, 255).astype(np.uint8)

    valid_pixels = scaled[interior].reshape(-1, 1)
    otsu, _ = cv.threshold(valid_pixels, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    relaxed = float(np.percentile(valid_pixels, RELAXED_PERCENTILE))
    threshold = min(otsu, relaxed)

    binary = ((scaled > threshold) & interior).astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (MORPHOLOGY_SIZE, MORPHOLOGY_SIZE))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    binary[~interior] = 0

    size = int(np.count_nonzero(interior))
    minimum = max(1, round(MIN_REGION_FRACTION * size))
    maximum = round(MAX_REGION_FRACTION * size)
    count, labels, statistics, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(binary)

    for label in range(1, count):
        area = int(statistics[label, cv.CC_STAT_AREA])

        if area >= minimum:
            result[labels == label] = 255

    return result


def overlay_contours(image: np.ndarray, binary: np.ndarray) -> np.ndarray:
    marked = image.copy()
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(marked, contours, -1, (0, 255, 255), 2)

    return marked


def run() -> None:
    project = Project.open()
    predictor = image_predictor(str(project.checkpoint))
    figure, axes = plt.subplots(3, len(HEAD_BOXES), figsize=(16, 12))

    for column, (image_id, (x1, y1, x2, y2)) in enumerate(HEAD_BOXES.items()):
        image = project.media.read_subject_image(SUBJECT, image_id)
        height, width = image.shape[:2]

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid head box for {SUBJECT}/{image_id}")

        mask = subject_mask(predictor, image, (x1, y1, x2, y2))
        crop = image[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2]
        enhanced, luminance = enhance(crop)
        binary = binary_pattern(luminance, valid)
        overlay = overlay_contours(enhanced, binary)

        region_count = cv.connectedComponents(binary)[0] - 1
        print(
            f"{SUBJECT}/{image_id}: SAM2={np.mean(valid):.1%} of crop, "
            f"pattern={np.count_nonzero(binary) / binary.size:.1%} of crop, "
            f"regions={region_count}"
        )

        panels = (
            (cv.cvtColor(enhanced, cv.COLOR_BGR2RGB), "CLAHE (head crop)", None),
            (binary, "Binary pattern: white = candidate marking", "gray"),
            (cv.cvtColor(overlay, cv.COLOR_BGR2RGB), "Pattern contours", None),
        )

        for row, (view, title, color_map) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(
                view, cmap=color_map, vmin=0 if color_map else None, vmax=255 if color_map else None
            )
            axis.set_title(f"{SUBJECT}/{image_id}: {title}")
            axis.set_axis_off()

    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


if __name__ == "__main__":
    run()
