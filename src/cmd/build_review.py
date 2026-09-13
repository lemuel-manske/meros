import csv

from typing import TypedDict

from src.artifacts import (
    get_vids,
    path_to_crop,
    path_to_review,
)
from src.metadata import (
    get_track_annotations,
    get_track_metadata,
)


class ReviewRow(TypedDict):
    video_id: str
    track_id: str
    frame_idx: int
    crop_path: str
    viewpoint: str
    quality: int
    track_correct: str
    viewpoint_correct: str
    head_pattern_visible: str
    body_pattern_visible: str
    usable_for_identity: str
    individual_id: str


_FIELDS = list(ReviewRow.__annotations__)


def space_frames(
    rows: list[ReviewRow],
    min_frame_gap: int,
) -> list[ReviewRow]:
    spaced = []

    for row in sorted(rows, key=lambda row: row["frame_idx"]):
        if not spaced or row["frame_idx"] - spaced[-1]["frame_idx"] >= min_frame_gap:
            spaced.append(row)

    return spaced


def get_track_intervals(
    track_id: str,
    annotations: list,
) -> list:
    intervals = [
        annotation
        for annotation in annotations
        if annotation["track_id"] == track_id
    ]

    return sorted(
        intervals,
        key=lambda annotation: annotation["start_frame"],
    )


def get_frame_annotation(
    frame_idx: int,
    intervals: list,
):
    return next(
        (
            interval
            for interval in intervals
            if interval["start_frame"] <= frame_idx <= interval["end_frame"]
        ),
        None,
    )


def select_group_frames(
    groups: dict[str, list[ReviewRow]],
    samples: int,
    min_frame_gap: int,
) -> list[ReviewRow]:
    return [
        row
        for viewpoint in sorted(groups)
        for row in select_frames(
            groups[viewpoint],
            samples,
            min_frame_gap,
        )
    ]


def build_review(
    samples: int = 3,
    min_frame_gap: int = 15,
) -> list[ReviewRow]:
    return [
        row
        for video_id in get_vids()
        for row in build_review_rows(
            video_id,
            samples,
            min_frame_gap,
        )
    ]


def validate_annotations(
    video_id: str,
    tracks: dict,
    annotations: list,
) -> None:
    for annotation in annotations:
        if annotation["track_id"] not in tracks:
            raise ValueError(
                f"{video_id}: annotations refer to unknown track "
                f"{annotation['track_id']}."
            )


def select_frames(
    rows: list[ReviewRow],
    samples: int,
    min_frame_gap: int,
) -> list[ReviewRow]:
    if samples < 1 or min_frame_gap < 1:
        raise ValueError("Sample count and minimum frame gap must be positive.")

    rows = space_frames(rows, min_frame_gap)

    if len(rows) <= samples:
        return rows

    if samples == 1:
        return [rows[len(rows) // 2]]

    return [
        rows[i * (len(rows) - 1) // (samples - 1)]
        for i in range(samples)
    ]


def build_track_review_rows(
    video_id: str,
    track_id: str,
    frames: dict,
    intervals: list,
    samples: int,
    min_frame_gap: int,
) -> list[ReviewRow]:
    validate_intervals(video_id, intervals)

    groups: dict[str, list[ReviewRow]] = {}

    for frame_idx in sorted(map(int, frames)):
        row = build_review_row(
            video_id,
            track_id,
            frame_idx,
            intervals,
        )

        if row is None:
            continue

        groups.setdefault(row["viewpoint"], []).append(row)

    selected = select_group_frames(
        groups,
        samples,
        min_frame_gap,
    )

    if not selected:
        raise ValueError(
            f"{video_id}, track {track_id}: "
            "no track crops available for review."
        )

    return sorted(selected, key=lambda row: row["frame_idx"])


def validate_intervals(
    video_id: str,
    intervals: list,
) -> None:
    previous_end = -1

    for interval in intervals:
        start = interval["start_frame"]
        end = interval["end_frame"]

        if start < 0 or end < start or start <= previous_end:
            raise ValueError(
                f"{video_id}: invalid or overlapping interval "
                f"{start}–{end}."
            )

        if not interval["viewpoint"].strip():
            raise ValueError(
                f"{video_id}: missing viewpoint for interval "
                f"{start}–{end}."
            )

        previous_end = end


def build_review_rows(
    video_id: str,
    samples: int = 3,
    min_frame_gap: int = 15,
) -> list[ReviewRow]:
    metadata = get_track_metadata(video_id)
    annotations = get_track_annotations(video_id)

    validate_annotations(video_id, metadata["tracks"], annotations)

    rows = []

    for track_id, track in metadata["tracks"].items():
        intervals = get_track_intervals(track_id, annotations)

        rows.extend(
            build_track_review_rows(
                video_id,
                track_id,
                track["frames"],
                intervals,
                samples,
                min_frame_gap,
            )
        )

    return rows


def build_review_row(
    video_id: str,
    track_id: str,
    frame_idx: int,
    intervals: list,
) -> ReviewRow | None:
    crop_path = path_to_crop(video_id, frame_idx, track_id)

    annotation = get_frame_annotation(frame_idx, intervals)

    return {
        "video_id": video_id,
        "track_id": track_id,
        "frame_idx": frame_idx,
        "crop_path": crop_path,
        "viewpoint": annotation["viewpoint"] if annotation else "",
        "quality": annotation["quality"] if annotation else -1,
        "track_correct": "",
        "viewpoint_correct": "",
        "head_pattern_visible": "",
        "body_pattern_visible": "",
        "usable_for_identity": "",
        "individual_id": "",
    }


def persist_review(rows: list[ReviewRow]) -> None:
    review_path = path_to_review()

    # exclusive creation protects existing human review work.
    with open(review_path, "x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    """
    Builds review samples for all videos and saves them to disk.

    Reviews are meant to be human audited for track correctness, viewpoint correctness, and other attributes.
    """

    rows = build_review()

    persist_review(rows)

    print(f"Saved {len(rows)} review samples to {path_to_review()}.")
