import argparse
import cv2 as cv
import numpy as np

from pathlib import Path


def positive_int(value: str) -> int:
    """
    Convert a string to a positive integer.
    """
    number = int(value)

    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")

    return number


def glob_jpgs(path: str) -> list[Path]:
    """
    Glob all .jpg files in the given path.
    """
    return list(Path(path).glob("*.jpg"))


def glob_csvs(path: str) -> list[Path]:
    """
    Glob all .csv files in the given path.
    """
    return list(Path(path).glob("*.csv"))


def gray_it(it: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a BGRA masked crop to grayscale and generate a mask for feature detection.
    """
    gray = cv.cvtColor(it[:, :, :3], cv.COLOR_BGR2GRAY)
    mask = (it[:, :, 3] == 255).astype(np.uint8) * 255

    # exclude the silhouette so matching favors internal markings.
    mask = cv.erode(mask, np.ones((21, 21), np.uint8))
    gray = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    return gray, mask
