"""
Compare spot-enhancement strategies over complete fish photographs.

Run from the repository root:
    python -m meros.sandbox.full_image_contrast_maps

Responses are exploratory, not verified spot detections. Nothing is saved to disk.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


IMAGES = (("rogerio", 9), ("rogerio", 12))
MAX_DISPLAY_WIDTH = 1000
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILES = (8, 8)

INNER_SIGMA = 2.0
OUTER_SIGMA = 8.0
BLACKHAT_SIZE = 13
TENSOR_SIGMA = 2.0
CANDIDATE_PERCENTILE = 91
MIN_CANDIDATE_AREA = 5
MAX_CANDIDATE_AREA = 180
MIN_CIRCULARITY = 0.2
MAX_ASPECT_RATIO = 3.5


def responses(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    light = cv.cvtColor(image, cv.COLOR_BGR2LAB)[:, :, 0]
    enhanced = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILES).apply(
        light
    )
    values = enhanced.astype(np.float32)

    fine = cv.GaussianBlur(values, (0, 0), sigmaX=INNER_SIGMA)
    broad = cv.GaussianBlur(values, (0, 0), sigmaX=OUTER_SIGMA)
    dark_dog = np.maximum(broad - fine, 0)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (BLACKHAT_SIZE, BLACKHAT_SIZE))
    blackhat = cv.morphologyEx(enhanced, cv.MORPH_BLACKHAT, kernel)

    laplacian = cv.Laplacian(fine, cv.CV_32F, ksize=3)
    dark_log = np.maximum(laplacian, 0)

    return enhanced, dark_dog, blackhat, dark_log


def isotropy(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    gradient_x = cv.Sobel(values, cv.CV_32F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(values, cv.CV_32F, 0, 1, ksize=3)

    xx = cv.GaussianBlur(gradient_x * gradient_x, (0, 0), sigmaX=TENSOR_SIGMA)
    yy = cv.GaussianBlur(gradient_y * gradient_y, (0, 0), sigmaX=TENSOR_SIGMA)
    xy = cv.GaussianBlur(gradient_x * gradient_y, (0, 0), sigmaX=TENSOR_SIGMA)

    difference = np.sqrt(np.maximum((xx - yy) ** 2 + 4 * xy**2, 0))
    coherence = difference / np.maximum(xx + yy, 1e-6)

    return np.clip(1 - coherence, 0, 1)


def candidates(response: np.ndarray) -> np.ndarray:
    positive = response[response > 0]
    result = np.zeros(response.shape, dtype=np.uint8)

    if positive.size == 0:
        return result

    threshold = np.percentile(positive, CANDIDATE_PERCENTILE)
    binary = (response >= threshold).astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    count, labels, statistics, _ = cv.connectedComponentsWithStats(binary, connectivity=8)

    for label in range(1, count):
        area = int(statistics[label, cv.CC_STAT_AREA])

        if not MIN_CANDIDATE_AREA <= area <= MAX_CANDIDATE_AREA:
            continue

        component = (labels == label).astype(np.uint8)
        contours, _ = cv.findContours(component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        contour = contours[0]
        perimeter = cv.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / perimeter**2
        _, _, width, height = cv.boundingRect(contour)
        aspect_ratio = max(width / height, height / width)

        if circularity >= MIN_CIRCULARITY and aspect_ratio <= MAX_ASPECT_RATIO:
            result[labels == label] = 255

    return result


def show(image: np.ndarray, name: str) -> None:
    height, width = image.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / width)
    resized = cv.resize(image, (round(width * scale), round(height * scale)))

    enhanced, dog, blackhat, log = responses(resized)
    spotness = dog * isotropy(enhanced)
    mask = candidates(spotness)

    overlay = cv.cvtColor(enhanced, cv.COLOR_GRAY2RGB)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(overlay, contours, -1, (255, 255, 0), 1)

    figure, axes = plt.subplots(3, 3, figsize=(17, 14))
    panels = (
        (cv.cvtColor(resized, cv.COLOR_BGR2RGB), "Original", None),
        (enhanced, "CLAHE luminance", "gray"),
        (dog, "DoG: dark structures", "gray"),
        (blackhat, "Black-hat: dark structures", "gray"),
        (log, "LoG: dark blob response", "gray"),
        (isotropy(enhanced), "Isotropy (less elongated)", "gray"),
        (spotness, "DoG dark × isotropy", "gray"),
        (mask, "Filtered spot candidates", "gray"),
        (overlay, f"Candidates overlaid ({len(contours)})", None),
    )

    for axis, (panel, title, palette) in zip(axes.flat, panels, strict=True):
        axis.imshow(panel, cmap=palette)
        axis.set_title(title)
        axis.set_axis_off()

    figure.suptitle(f"{name}: exploratory full-image spot enhancement")
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
