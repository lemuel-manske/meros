import cv2 as cv

from src.artifacts import path_to_frame
from src.frame import get_frames

from src.meros.dataset import videos, Video


def extract_n_persist_frames(vid: Video) -> None:
    vid_id = vid.video_id

    frames = get_frames(vid.path)

    for i in range(len(frames)):
        frame_path = path_to_frame(vid_id, i)

        cv.imwrite(frame_path, frames[i])

        print(f"Frame {i} saved to {frame_path}")


if __name__ == "__main__":
    """
    Extracts frames from all videos in the specified paths and saves them to disk.
    """

    for vid in videos:
        extract_n_persist_frames(vid)

    print("Frames extracted and saved successfully.")
