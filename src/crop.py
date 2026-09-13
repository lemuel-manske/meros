import cv2 as cv
import cv2.typing as cvt

from src.artifacts import path_to_crop


def persist_crops(
    vid_id: str,
    crops: list[tuple[int, cvt.MatLike]],
    track_id: str = "1",
) -> None:
    """
    Persist track crops using their original video frame indices.
    """
    for frame_idx, crop in crops:
        crop_path = path_to_crop(vid_id, frame_idx, track_id)
        cv.imwrite(crop_path, crop)
