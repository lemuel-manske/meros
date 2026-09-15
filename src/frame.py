import cv2 as cv
import cv2.typing as cvt

from src.artifacts import path_to_frame


def get_frame(vid_id: str, frame_id: int) -> cvt.MatLike | None:
    """
    Get a frame from a video given its ID and frame index.
    """
    import os

    frame_path = path_to_frame(vid_id, frame_id)

    if not os.path.exists(frame_path):
        print(f"Warning: Frame {frame_id} not found for video {vid_id}.")

        return None

    return cv.imread(frame_path)


def get_frames(vid_id: str) -> list[cvt.MatLike]:
    """
    Extract frames from a video given its ID.
    """
    cap = cv.VideoCapture(vid_id)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames
