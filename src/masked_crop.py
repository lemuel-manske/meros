import cv2 as cv
import numpy as np

from src.artifacts import path_to_masked_crop


def read_masked_crop(
    vid_id: str,
    track_id: str,
    frame_idx: int,
) -> np.ndarray:
    """
    Read a BGRA masked crop from disk.
    """
    path = path_to_masked_crop(vid_id, frame_idx, track_id)

    crop = cv.imread(path, cv.IMREAD_UNCHANGED)

    if crop is None or crop.ndim != 3 or crop.shape[2] != 4:
        raise ValueError(f"Expected a BGRA masked crop: {path}.")

    return crop
