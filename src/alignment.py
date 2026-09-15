import cv2 as cv
import json
import numpy as np

from pathlib import Path
from typing import (
    Literal,
    NotRequired,
    TypedDict,
)

from src.artifacts import (
    path_to_aligned_crop,
    path_to_composite,
    path_to_alignment_media,
    path_to_alignment_metadata,
)


AlignmentStatus = Literal[
    "insufficient_matches",
    "estimation_failed",
    "candidate",
    "rejected_geometry",
]


class AlignmentMetadata(TypedDict):
    status: AlignmentStatus
    matches: int
    inliers: int
    frame_idx: NotRequired[int]
    inlier_ratio: NotRequired[float]
    median_error_px: NotRequired[float]
    inlier_hull_fraction: NotRequired[float]
    source_to_reference: NotRequired[list[list[float]]]


def persist_overlay(
    vid_id: str,
    track_id: str,
    frame_idx: int,
    overlay: np.ndarray,
) -> None:
    """
    Persist an overlay image to a PNG file.
    """
    parent = Path(path_to_alignment_media(vid_id, track_id))

    path = parent / f"{frame_idx}_overlay.png"

    if not cv.imwrite(str(path), overlay):
        raise RuntimeError(f"Could not save {path}.")


def persist_aligned_crop(
    vid_id: str,
    track_id: str,
    frame_idx: int,
    aligned_crop: np.ndarray,
) -> None:
    """
    Persist an aligned crop image to a PNG file.
    """
    parent = Path(path_to_alignment_media(vid_id, track_id))

    path = parent / f"{frame_idx}.png"

    if not cv.imwrite(str(path), aligned_crop):
        raise RuntimeError(f"Could not save {path}.")


def remove_alignment_images(
    vid_id: str,
    track_id: str,
    frame_idx: int,
) -> None:
    """
    Remove alignment images for a given frame index.
    """
    parent = Path(path_to_alignment_media(vid_id, track_id))

    for name in (f"{frame_idx}.png", f"{frame_idx}_overlay.png"):
        (parent / name).unlink(missing_ok=True)


def persist_alignment_metadata(
    vid_id: str,
    track_id: str,
    start_frame: int,
    end_frame: int,
    reference_frame: int,
    rows: list[AlignmentMetadata],
) -> None:
    """
    Persist alignment metadata to a JSON file.
    """
    path = path_to_alignment_metadata(vid_id, track_id)

    with open(path, "w") as f:
        json.dump(
            {
                "video_id": vid_id,
                "track_id": track_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "reference_frame": reference_frame,
                "method": "SIFT mutual ratio matches, affine RANSAC",
                "frames": rows,
            },
            f,
            indent=2,
        )

        f.write("\n")


def read_alignment_metadata(
    vid_id: str,
    track_id: str,
) -> dict:
    """
    Read alignment metadata from a JSON file.
    """
    path = path_to_alignment_metadata(vid_id, track_id)

    with open(path) as f:
        return json.load(f)


def read_aligned_crop(
    vid_id: str,
    track_id: str,
    frame_idx: int,
) -> np.ndarray:
    """
    Read an aligned crop image from a PNG file.
    """
    path = path_to_aligned_crop(vid_id, track_id, frame_idx)

    crop = cv.imread(path, cv.IMREAD_UNCHANGED)

    if crop is None or crop.ndim != 3 or crop.shape[2] != 4:
        raise ValueError(f"Expected a BGRA aligned crop: {path}.")

    return crop


def persist_composite(
    vid_id: str,
    track_id: str,
    composite: np.ndarray,
) -> None:
    """
    Persist a composite image to a PNG file.
    """
    path = path_to_composite(vid_id, track_id)

    if not cv.imwrite(path, composite):
        raise RuntimeError(f"Could not save {path}.")
