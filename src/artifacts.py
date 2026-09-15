from src.consts import (
    ALIGNMENT_MEDIA_FOLDER,
    ALIGNMENT_METADATA_FOLDER,
    ANNOTATIONS_FOLDER,
    CROPS_FOLDER,
    FRAMES_FOLDER,
    MASKED_CROPS_FOLDER,
    REVIEWS_FOLDER,
    TRACKS_FOLDER,
    VIDEOS_FOLDER,
    VISUALIZATIONS_FOLDER,
)
from src.fn import (
    glob_csvs,
    glob_jpgs,
)


# collection for all vids intended to be processed
VIDEOS = [
    "102_13.mov",
    "25_1.mov",
    "26_4.mov",
    "65_5.mov",
    "65_6.mov",
    "65_8.mov",
]


_REVIEW_FILE_PATTERN = "%d_%m_%Y"  # DD_MM_YYYY


def path_to_alignment_media(vid_id: str, track_id: str) -> str:
    """
    Get the path to the alignment media for a given video ID and track ID.
    """
    import os

    dir_path = f"{ALIGNMENT_MEDIA_FOLDER}/{vid_id}/{track_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def path_to_alignment_metadata(vid_id: str, track_id: str) -> str:
    """
    Get the path to the alignment metadata for a given video ID and track ID.
    """
    import os

    dir_path = f"{ALIGNMENT_METADATA_FOLDER}/{vid_id}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return f"{dir_path}/{track_id}.json"


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


def path_to_review() -> str:
    """
    Get the path to the review.
    """
    from datetime import datetime

    fname = datetime.now().strftime(_REVIEW_FILE_PATTERN)

    return f"{REVIEWS_FOLDER}/{fname}.csv"


def path_to_vid(vid_id):
    """
    Get the path to a video given its ID.
    """
    return f"{VIDEOS_FOLDER}/{vid_id}"


def get_review_paths() -> list[str]:
    """
    Get the paths to all review files, sorted by date.
    """
    from datetime import datetime

    return [str(p) for p in sorted(
        glob_csvs(REVIEWS_FOLDER),
        key=lambda path: datetime.strptime(path.stem, _REVIEW_FILE_PATTERN),
    )]


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


def path_to_aligned_crop(
    vid_id: str,
    track_id: str,
    frame_idx: int,
) -> str:
    """
    Get the path to an aligned crop for a given video ID, track ID, and frame index.
    """
    return f"{path_to_alignment_media(vid_id, track_id)}/{frame_idx}.png"


def path_to_composite(
    vid_id: str,
    track_id: str,
) -> str:
    """
    Get the path to the composite image for a given video ID and track ID.
    """
    return f"{path_to_alignment_media(vid_id, track_id)}/median.png"


def path_to_enhanced_composite(
    vid_id: str,
    track_id: str,
) -> str:
    """
    Get the path to the enhanced composite image for a given video ID and track ID.
    """
    return f"{path_to_alignment_media(vid_id, track_id)}/median_contrast.png"


def path_to_composite_comparison(
    vid_id: str,
    track_id: str,
) -> str:
    """
    Get the path to the composite comparison image for a given video ID and track ID.
    """
    return f"{path_to_alignment_media(vid_id, track_id)}/median_comparison.png"
