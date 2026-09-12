import cv2 as cv
import cv2.typing as cvt

from src.artifacts import path_to_track


def persist_tracks(vid_id: str, frames: list[tuple[int, cvt.MatLike]]) -> None:
    """
    Persist track crops using their original video frame indices.
    """
    for frame_idx, frame in frames:
        track_path = path_to_track(vid_id, frame_idx)
        cv.imwrite(track_path, frame)
