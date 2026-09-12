import json

from typing import TypedDict

from src.artifacts import path_to_metadata


class VisMetadataObjectFrame(TypedDict):
    bbox: list[int]
    mask_area: int


class VisMetadataObject(TypedDict):
    frames: dict[str, VisMetadataObjectFrame]


class VisMetadata(TypedDict):
    video_id: str
    initial_frame: int
    initial_bbox: list[int]
    objects: dict[str, VisMetadataObject]


def persist_vis_metadata(vid_id: str, metadata: dict) -> None:
    """
    Persist the metadata for a given video ID.
    """
    metadata_path = path_to_metadata(vid_id)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def get_vis_metadata(vid_id: str) -> VisMetadata:
    """
    Get the metadata for a given video ID.
    """
    metadata_path = path_to_metadata(vid_id)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata
