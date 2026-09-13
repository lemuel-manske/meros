import csv
import json

from typing import NotRequired, TypedDict

from src.artifacts import path_to_track_annotations, path_to_track_metadata


class TrackObservation(TypedDict):
    bbox: list[int]
    mask_area: int


class Track(TypedDict):
    initial_frame: NotRequired[int]
    initial_bbox: NotRequired[list[int]]
    frames: dict[str, TrackObservation]


class TrackMetadata(TypedDict):
    video_id: str
    initial_frame: NotRequired[int]
    initial_bbox: NotRequired[list[int]]
    tracks: dict[str, Track]


class TrackAnnotation(TypedDict):
    track_id: str
    start_frame: int
    end_frame: int
    viewpoint: str
    quality: int


def persist_track_metadata(vid_id: str, metadata: TrackMetadata) -> None:
    """
    Persist the metadata for a given video ID.
    """
    metadata_path = path_to_track_metadata(vid_id)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_track_metadata(vid_id: str) -> TrackMetadata:
    """
    Get the metadata for a given video ID.
    """
    metadata_path = path_to_track_metadata(vid_id)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


def get_track_annotations(vid_id: str) -> list[TrackAnnotation]:
    """
    Get human viewpoint and quality annotations for tracks in a video.
    """
    metadata_path = path_to_track_annotations(vid_id)

    with open(metadata_path, "r") as f:
        metadata = csv.DictReader(f)

        return [{
            "track_id": row.get("track_id", "1"),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "viewpoint": row["viewpoint"],
            "quality": int(row["quality"])
        } for row in metadata]
