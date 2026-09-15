import csv

from typing import TypedDict

from src.artifacts import paths_to_reviews, path_to_review


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


_FIELD_NAMES = list(ReviewRow.__annotations__)


def get_review_reader(path: str) -> csv.DictReader:
    """
    Get a CSV reader for the review file at the given path.
    """
    with open(path, newline="", encoding="utf-8") as source:
        return csv.DictReader(source)


def get_review_writer(path: str) -> csv.DictWriter:
    """
    Get a CSV writer for the review file at the given path.
    """
    output = open(path, "w", newline="", encoding="utf-8")

    return csv.DictWriter(
        output,
        fieldnames=_FIELD_NAMES,
    )


def persist_review(rows: list[ReviewRow]) -> str:
    """
    Persist the given review rows to the specified path.
    """
    path = path_to_review()

    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=_FIELD_NAMES,
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def get_review_dataset() -> list[ReviewRow]:
    """
    Load all dated reviews into memory without changing the source files.

    Prioritize the most recent review for each (video_id, track_id, frame_idx) combination.
    """
    paths = paths_to_reviews()

    observations = {}

    for path in paths:
        reader = get_review_reader(path)

        missing = set(_FIELD_NAMES) - set(reader.fieldnames or [])

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
