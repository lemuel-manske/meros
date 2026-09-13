import csv

from datetime import datetime
from pathlib import Path
from typing import TypedDict

from src.artifacts import path_to_reviews


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


def get_review_dataset() -> list[ReviewRow]:
    """
    Load all dated reviews into memory without changing the source files.

    Read reviews/DD_MM_YYYY.csv from oldest to newest by filename date.
    The newest row replaces the entire older row for the same
    (video_id, track_id, frame_idx). Other observations of the same track
    or individual remain; individual_id is a label, not a deduplication key.
    Within one file, the last row for an observation wins.

    Return rows sorted by video ID, track ID, and numeric frame index.
    Parse frame_idx and quality as integers; blank quality becomes -1.
    Preserve all decisions, including uncertain, unusable, and blank labels.
    Return an empty list when no review files exist. Invalid dates or
    missing columns raise ValueError rather than silently dropping a file.
    """
    paths = sorted(
        Path(path_to_reviews()).glob("*.csv"),
        key=lambda path: datetime.strptime(path.stem, "%d_%m_%Y"),
    )

    observations = {}

    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            missing = set(ReviewRow.__annotations__) - set(reader.fieldnames or [])

            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}.")

            for row in reader:
                observation = {
                    field: row[field]
                    for field in ReviewRow.__annotations__
                }

                observation["frame_idx"] = int(row["frame_idx"])
                observation["quality"] = int(row["quality"]) if row["quality"] else -1

                key = (
                    observation["video_id"],
                    observation["track_id"],
                    observation["frame_idx"],
                )

                observations[key] = observation

    return [observations[key] for key in sorted(observations)]
