import cv2 as cv
import cv2.typing as cvt

from src.artifacts import path_to_track


def persist_tracks(vid_id: str, frames: list[cvt.MatLike]) -> None:
    """
    Persist the tracks for a given video ID.
    """
    for frame_idx, frame in enumerate(frames):
        track_path = path_to_track(vid_id, frame_idx)
        cv.imwrite(track_path, frame)
