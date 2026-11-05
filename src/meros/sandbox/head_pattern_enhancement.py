"""
Reproduce the deterministic head-pattern enhancements for Rogerio/9 and Rogerio/12.

From the repository root, after `make data-pull`:
    python -m meros.sandbox.head_pattern_enhancement

This experiment enhances recorded image information; it cannot recover
markings that were never captured or prove that two patterns are identical.
"""

from pathlib import Path

import cv2 as cv
import numpy as np

from meros.project import Project


SUBJECT = "rogerio"
CROPS = {
    9: (270, 118, 962, 616),
    12: (165, 115, 1680, 1120),
}
OUTPUT_DIR = Path("results/head-pattern-enhancement")
WIDTH = 900
HEIGHT = 630

LABELS = (
    "Original (head crop)",
    "LAB + CLAHE (color)",
    "Illumination correction + CLAHE",
    "CLAHE luminance",
    "Multiscale DoG (3-55 px)",
    "Coarse DoG (15-55 px)",
)

OUTPUT_VARIANTS = {
    "original": 0,
    "clahe": 1,
    "illumination": 2,
    "dog": 4,
    "coarse_dog": 5,
}


def uint8_percentile(image: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """
    Scale an image to uint8 using percentile limits for visualization.
    """
    low, high = np.percentile(image, (lower, upper))
    return np.clip((image - low) * 255 / max(high - low, 1e-6), 0, 255).astype(np.uint8)


def fit_canvas(image: np.ndarray) -> np.ndarray:
    """
    Fit a head crop into the original fixed-size comparison canvas.
    """
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


def process(image: np.ndarray) -> list[np.ndarray]:
    """
    Apply the six fixed representations from the initial head-pattern comparison.
    """
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    light, channel_a, channel_b = cv.split(lab)

    equalized = cv.createCLAHE(clipLimit=2.0, tileGridSize=(9, 7)).apply(light)
    clahe_color = cv.cvtColor(cv.merge((equalized, channel_a, channel_b)), cv.COLOR_LAB2BGR)

    illumination = cv.GaussianBlur(light.astype(np.float32), (0, 0), sigmaX=35)
    corrected = light.astype(np.float32) - illumination + float(np.median(illumination))
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected = cv.createCLAHE(clipLimit=1.8, tileGridSize=(9, 7)).apply(corrected)
    corrected_color = cv.cvtColor(
        cv.merge((corrected, channel_a, channel_b)), cv.COLOR_LAB2BGR
    )

    smooth = cv.bilateralFilter(light, 9, 26, 26).astype(np.float32)
    dog_small = cv.GaussianBlur(smooth, (0, 0), sigmaX=3) - cv.GaussianBlur(
        smooth, (0, 0), sigmaX=15
    )
    dog_large = cv.GaussianBlur(smooth, (0, 0), sigmaX=15) - cv.GaussianBlur(
        smooth, (0, 0), sigmaX=55
    )
    combined = 0.35 * dog_small + 0.65 * dog_large
    dog_view = cv.cvtColor(uint8_percentile(combined, 2, 98), cv.COLOR_GRAY2BGR)
    coarse_view = cv.cvtColor(uint8_percentile(dog_large, 2, 98), cv.COLOR_GRAY2BGR)
    gray_view = cv.cvtColor(
        cv.createCLAHE(clipLimit=2.2, tileGridSize=(9, 7)).apply(light), cv.COLOR_GRAY2BGR
    )

    return [image, clahe_color, corrected_color, gray_view, dog_view, coarse_view]


def comparison_sheet(images: dict[int, list[np.ndarray]]) -> np.ndarray:
    """
    Place both photographs side by side for each representation.
    """
    margin = 24
    heading = 90
    cell_height = HEIGHT + 40
    sheet = np.full(
        (heading + len(LABELS) * cell_height + margin, margin * 3 + WIDTH * 2, 3),
        (24, 26, 31),
        dtype=np.uint8,
    )

    font = cv.FONT_HERSHEY_SIMPLEX

    for column, image_id in enumerate(CROPS):
        position = (margin + column * (WIDTH + margin), 54)
        cv.putText(
            sheet, f"Rogerio / {image_id}", position, font, 1.2, (240, 240, 240), 2, cv.LINE_AA
        )

    for row, label in enumerate(LABELS):
        y = heading + row * cell_height
        cv.putText(
            sheet, label, (margin, y + 27), font, 0.75, (225, 225, 225), 2, cv.LINE_AA
        )

        for column, image_id in enumerate(CROPS):
            x = margin + column * (WIDTH + margin)
            sheet[y + 34 : y + 34 + HEIGHT, x : x + WIDTH] = images[image_id][row]

    return sheet


def save_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    """
    Persist one diagnostic image in the dedicated experiment results directory.
    """
    if not cv.imwrite(str(path), image, params or []):
        raise OSError(f"Could not write image: {path}")


def run() -> None:
    """
    Load archived photos and reproduce the diagnostic images.
    """
    project = Project.open()
    images: dict[int, list[np.ndarray]] = {}

    for image_id, (x1, y1, x2, y2) in CROPS.items():
        image = project.media.read_subject_image(SUBJECT, image_id)
        height, width = image.shape[:2]

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Crop for {SUBJECT}/{image_id} exceeds source image bounds")

        images[image_id] = process(fit_canvas(image[y1:y2, x1:x2]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_id, variants in images.items():
        for name, index in OUTPUT_VARIANTS.items():
            save_image(OUTPUT_DIR / f"rogerio_{image_id}_{name}.png", variants[index])

    path = OUTPUT_DIR / "rogerio_9_12_pattern_comparison.jpg"
    sheet = comparison_sheet(images)
    save_image(path, sheet, [cv.IMWRITE_JPEG_QUALITY, 94])
    print(f"Saved {path} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    run()
