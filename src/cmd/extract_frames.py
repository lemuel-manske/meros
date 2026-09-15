import cv2 as cv

from src.artifacts import (
    get_vid_id,
    paths_to_vids,
    path_to_frame,
)
from src.frame import get_frames


def extract_n_persist_frames(vid_raw_path: str) -> None:
    vid_id = get_vid_id(vid_raw_path)
    frames = get_frames(vid_raw_path)

    for i in range(len(frames)):
        frame_path = path_to_frame(vid_id, i)

        cv.imwrite(frame_path, frames[i])

        print(f"Frame {i} saved to {frame_path}")


if __name__ == "__main__":
    """
    Extracts frames from all videos in the specified paths and saves them to disk.
    """

    for vid_path in paths_to_vids():
        extract_n_persist_frames(vid_path)

    print("Frames extracted and saved successfully.")
