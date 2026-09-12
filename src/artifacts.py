from src.consts import FRAMES_FOLDER, VIS_FOLDER, VIDEOS_FOLDER


VIDEOS = [
    "260113_S3_SC_Monoboia_0025_1#.mov",
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


def path_to_vis(vid_id: str,frame_idx: int) -> str:
    """
    Get the path to a visualization given its index.
    """
    import os

    dir_path = f"{VIS_FOLDER}/{vid_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return f"{dir_path}/{frame_idx}.jpg"


def path_to_vid(vid_id):
    """
    Get the path to a video given its ID.
    """
    return f"{VIDEOS_FOLDER}/{vid_id}"


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
