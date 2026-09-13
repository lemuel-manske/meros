import csv

from pathlib import Path
from typing import TypedDict

from src.artifacts import (
    get_vids,
    path_to_review_manifest,
    path_to_track,
)
from src.metadata import (
    get_track_metadata,
    get_vis_metadata,
)

class ReviewRow(TypedDict):
    video_id: str
    track_id: str
    frame_idx: int
    crop_path: str
    viewpoint: str
    quality: int


FIELDS = list(ReviewRow.__annotations__)


def select_frames(
    rows: list[ReviewRow],
    samples: int,
    min_frame_gap: int,
) -> list[ReviewRow]:
    if samples < 1 or min_frame_gap < 1:
        raise ValueError("Sample count and minimum frame gap must be positive.")

    spaced: list[ReviewRow] = []

    for row in sorted(rows, key=lambda row: row["frame_idx"]):
        if not spaced or row["frame_idx"] - spaced[-1]["frame_idx"] >= min_frame_gap:
            spaced.append(row)

    if len(spaced) <= samples:
        return spaced

    if samples == 1:
        return [spaced[len(spaced) // 2]]

    return [
        spaced[i * (len(spaced) - 1) // (samples - 1)]
        for i in range(samples)
    ]


def build_review_rows(
    video_id: str,
    samples: int = 3,
    min_frame_gap: int = 15,
) -> list[ReviewRow]:
    metadata = get_vis_metadata(video_id)

    # the current crop extractor saves only object 1 in a per-video folder.
    # refuse multi-object metadata because interval labels have no track id.

    if set(metadata["objects"]) != {"1"}:
        raise ValueError(f"{video_id}: expected a single track with object ID 1.")

    intervals = sorted(get_track_metadata(video_id), key=lambda row: row["start_frame"])

    previous_end = -1

    for interval in intervals:
        start, end = interval["start_frame"], interval["end_frame"]

        if start < 0 or end < start or start <= previous_end:
            raise ValueError(f"{video_id}: invalid or overlapping interval {start}–{end}.")

        if not interval["viewpoint"].strip():
            raise ValueError(f"{video_id}: missing viewpoint for interval {start}–{end}.")

        previous_end = end

    groups = {}

    for frame_idx in sorted(map(int, metadata["objects"]["1"]["frames"])):
        annotation = next(
            (row for row in intervals if row["start_frame"] <= frame_idx <= row["end_frame"]),
            None,
        )

        if annotation is None:
            continue

        crop_path = Path(path_to_track(video_id, frame_idx))

        if not crop_path.is_file():
            continue

        row = {
            "video_id": video_id,
            "track_id": "1",
            "frame_idx": frame_idx,
            "crop_path": str(crop_path),
            "viewpoint": annotation["viewpoint"],
            "quality": annotation["quality"],
        }

        groups.setdefault(row["viewpoint"], []).append(row)

    selected = [
        row
        for viewpoint in sorted(groups)
        for row in select_frames(groups[viewpoint], samples, min_frame_gap)
    ]

    if not selected:
        raise ValueError(f"{video_id}: no annotated track crops available for review.")

    return sorted(selected, key=lambda row: row["frame_idx"])


def persist_review_manifest(rows: list[ReviewRow]) -> None:
    manifest_path = path_to_review_manifest()

    # exclusive creation protects any human review work in an existing csv.
    with open(manifest_path, "x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    """
    Build a review manifest for all videos in the specified paths and save it to disk.
    """
    rows = [
        row
        for video_id in get_vids()
        for row in build_review_rows(video_id)
    ]

    persist_review_manifest(rows)

    print(f"Saved {len(rows)} review samples to {path_to_review_manifest()}.")
