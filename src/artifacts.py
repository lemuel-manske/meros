from src.consts import (
    ANNOTATIONS_FOLDER,
    CROPS_FOLDER,
    MASKED_CROPS_FOLDER,
    FRAMES_FOLDER,
    TRACKS_FOLDER,
    VIDEOS_FOLDER,
    VISUALIZATIONS_FOLDER,
)
from src.fn import glob_jpgs


VIDEOS = [
    "102_13.mov",
    "25_1.mov",
    "26_4.mov",
    "65_5.mov",
    "65_6.mov",
    "65_8.mov",
]


def path_to_frame(vid_id: str, frame_idx: int) -> str:
    """
    Get the path to a frame given its index.
    """
    import os

    dir_path = f"{FRAMES_FOLDER}/{vid_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return f"{dir_path}/{frame_idx}.jpg"


def path_to_frames(vid_id: str) -> str:
    """
    Get the path to the frames folder for a given video ID.
    """
    return f"{FRAMES_FOLDER}/{vid_id}"


def path_to_visualization(vid_id: str,frame_idx: int) -> str:
    """
    Get the path to a visualization given its index.
    """
    import os

    dir_path = f"{VISUALIZATIONS_FOLDER}/{vid_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return f"{dir_path}/{frame_idx}.jpg"


def path_to_track_metadata(vid_id: str) -> str:
    """
    Get the path to the metadata file for a given video ID.
    """
    import os

    dir_path = f"{TRACKS_FOLDER}/{vid_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return f"{dir_path}/metadata.json"


def path_to_track_annotations(vid_id: str) -> str:
    """
    Get the path to the track annotations file for a given video ID.
    """
    import os

    dir_path = f"{ANNOTATIONS_FOLDER}/{vid_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return f"{dir_path}/metadata.csv"


def path_to_crop(vid_id: str, frame_idx: int, track_id: str = "1") -> str:
    """
    Get the path to a crop for a track and source frame.
    """
    import os

    dir_path = f"{CROPS_FOLDER}/{vid_id}/{track_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return f"{dir_path}/{frame_idx}.jpg"


def path_to_masked_crop(
    vid_id: str,
    frame_idx: int,
    track_id: str,
) -> str:
    """
    Get the PNG path for a crop with a transparent background.
    """
    import os

    dir_path = f"{MASKED_CROPS_FOLDER}/{vid_id}/{track_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return f"{dir_path}/{frame_idx}.png"


def path_to_reviews() -> str:
    """
    Get the folder containing dated human reviews.
    """
    return "reviews"


def path_to_review() -> str:
    """
    Get the path to the review.
    """
    return "review.csv"


def path_to_vid(vid_id):
    """
    Get the path to a video given its ID.
    """
    return f"{VIDEOS_FOLDER}/{vid_id}"


def get_frame_ids(vid_id: str) -> list[int]:
    """
    Count the number of frames for a given video ID.
    """
    return sorted(int(path.stem) for path in glob_jpgs(path_to_frames(vid_id)))


def get_vid_id(vid_raw_path):
    """
    Extract the video ID from the video path.
    """
    return vid_raw_path.split("/")[-1].split(".")[0]


def get_vids_paths() -> list[str]:
    """
    Get the paths to all videos in the VIDEOS list.
    """
    return [f"{VIDEOS_FOLDER}/{vid}" for vid in VIDEOS]


def get_vids() -> list[str]:
    """
    Get the list of video IDs.
    """
    return [get_vid_id(vid) for vid in VIDEOS]
