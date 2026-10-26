from pathlib import Path

import cv2 as cv
import numpy as np


def valid_image(
    path: Path,
    channels: int,
    shape: tuple[int, int] | None = None,
    *,
    mask_area: int | None = None,
) -> bool:
    if not path.is_file():
        return False

    try:
        image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    except cv.error:
        return False

    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != channels:
        return False

    if shape is not None and image.shape[:2] != shape:
        return False

    if channels == 4:
        alpha = image[:, :, 3]

        if not np.all((alpha == 0) | (alpha == 255)) or not np.any(alpha == 255):
            return False

        if mask_area is not None and np.count_nonzero(alpha) != mask_area:
            return False

    return True


def read_image(path: Path, flags: int = cv.IMREAD_COLOR) -> np.ndarray:
    image = cv.imread(str(path), flags)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")
