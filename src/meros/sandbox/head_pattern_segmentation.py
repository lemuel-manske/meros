"""
Segment candidate light and dark skin patches in Rogerio/9 and Rogerio/12.

From the repository root:
    make install-sam2
    python -m meros.sandbox.head_pattern_segmentation

Uses the official SAM2 image predictor with a head-region box prompt.
The box is not a ground-truth mask: review the SAM2 mask before trusting
the CLAHE/tonal segmentations. Shows plots only; writes no files.
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
BACKGROUND_SIGMA = 22.0
RESIDUAL_THRESHOLD = 12.0
MIN_REGION_AREA = 45
MAX_REGION_AREA_FRACTION = 0.08
MASK_EDGE_MARGIN = 8


def image_predictor(checkpoint: str) -> Any:
    """
    Build SAM2's still-image predictor instead of the existing video predictor.
    """
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
    """
    Segment the head-prompted fish region using the unmodified full photograph.
    """
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    masks, _, _ = predictor.predict(
        box=np.asarray(box, dtype=np.float32),
        multimask_output=False,
    )

    return np.asarray(masks[0], dtype=bool)


def enhance(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Equalize LAB luminance without altering the chroma channels.
    """
    luminance, channel_a, channel_b = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))
    equalized = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    light = equalized.apply(luminance)
    color = cv.cvtColor(cv.merge((light, channel_a, channel_b)), cv.COLOR_LAB2BGR)

    return color, light


def segment_tonal_regions(
    luminance: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local light/dark deviations within the masked fish skin.

    Validity is eroded to suppress artificial mask edges. The outputs
    are exploratory tonal regions, not verified biological markings.
    """
    background = cv.GaussianBlur(
        luminance.astype(np.float32),
        (0, 0),
        sigmaX=BACKGROUND_SIGMA,
    )
    residual = luminance.astype(np.float32) - background
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    interior = cv.erode(
        (valid.astype(np.uint8) * 255),
        cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (2 * MASK_EDGE_MARGIN + 1, 2 * MASK_EDGE_MARGIN + 1)
        ),
    ) > 0

    minimum = MIN_REGION_AREA
    maximum = int(MAX_REGION_AREA_FRACTION * np.count_nonzero(interior))
    masks = []

    for selected in (residual >= RESIDUAL_THRESHOLD, residual <= -RESIDUAL_THRESHOLD):
        binary = (selected & interior).astype(np.uint8) * 255
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

        count, labels, statistics, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
        kept = np.zeros_like(binary)

        for label in range(1, count):
            area = int(statistics[label, cv.CC_STAT_AREA])

            if minimum <= area <= maximum:
                kept[labels == label] = 255

        masks.append(kept)

    return masks[0], masks[1]


def draw_regions(image: np.ndarray, bright: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    Overlay bright and dark region outlines without replacing original pixels.
    """
    overlay = image.copy()

    for mask, color in ((bright, (0, 255, 255)), (dark, (255, 120, 40))):
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(overlay, contours, -1, color, 2)

    return overlay


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
        region = image[y1:y2, x1:x2]
        valid = mask[y1:y2, x1:x2]
        enhanced, light = enhance(region)
        bright, dark = segment_tonal_regions(light, valid)
        overlay = draw_regions(enhanced, bright, dark)

        boundary = enhanced.copy()
        boundary[~valid] = (20, 20, 20)
        print(
            f"{SUBJECT}/{image_id}: mask={np.mean(valid):.1%} of crop, "
            f"bright={cv.connectedComponents(bright)[0] - 1}, "
            f"dark={cv.connectedComponents(dark)[0] - 1} regions"
        )

        panels = (
            (cv.cvtColor(region, cv.COLOR_BGR2RGB), "Original head crop"),
            (cv.cvtColor(boundary, cv.COLOR_BGR2RGB), "SAM2 mask + CLAHE"),
            (cv.cvtColor(overlay, cv.COLOR_BGR2RGB), "Light / dark tonal contours"),
        )

        for row, (panel, title) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(panel)
            axis.set_title(f"{SUBJECT}/{image_id}: {title}")
            axis.set_axis_off()

    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


if __name__ == "__main__":
    run()
