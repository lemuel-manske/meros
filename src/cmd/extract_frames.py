import cv2 as cv

from src.artifacts import (
    get_vid_id,
    get_vids_paths,
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


def extract_n_persist_frames_for_all_vids(vid_paths: list[str]) -> None:
    for vid_path in vid_paths:
        extract_n_persist_frames(vid_path)


if __name__ == "__main__":
    """
    Extracts frames from all videos in the specified paths and saves them to disk.
    """

    extract_n_persist_frames_for_all_vids(get_vids_paths())

    print("Frames extracted and saved successfully.")
