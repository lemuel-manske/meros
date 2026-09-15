import json

from src.consts import VIDEOS_METADATA_FOLDER


class Video:
    video_id: str
    fname: str
    path: str


def _read() -> list[dict]:
    with open(f"{VIDEOS_METADATA_FOLDER}/videos.json", "r") as f:
        return json.load(f)


def _cast(video_metadata: dict) -> Video:
    video = Video()
    video.video_id = video_metadata["video_id"]
    video.fname = video_metadata["fname"]

    video.path = f"{VIDEOS_METADATA_FOLDER}/{video.fname}"

    return video


def get_videos() -> list[Video]:
    """
    Get the list of videos from the metadata file.
    """
    videos_metadata = _read()

    return [
        _cast(video_metadata) for video_metadata in videos_metadata
    ]
