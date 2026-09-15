import json

from dataclasses import dataclass

from src.consts import INDIVIDUALS_METADATA_FOLDER


@dataclass
class IndividualTrack:
    video_id: str
    track_id: str
    start_frame: int
    end_frame: int
    reference_frame: int


@dataclass
class Individual:
    individual_id: str
    tracks: list[IndividualTrack]


def _read() -> list[dict]:
    with open(f"{INDIVIDUALS_METADATA_FOLDER}/individuals.json", "r") as f:
        return json.load(f)


def _cast_track(track_metadata: dict) -> IndividualTrack:
    video_id = track_metadata["video_id"]
    track_id = track_metadata["track_id"]
    start_frame = track_metadata["start_frame"]
    end_frame = track_metadata["end_frame"]
    reference_frame = track_metadata["reference_frame"]

    track = IndividualTrack(
        video_id=video_id,
        track_id=track_id,
        start_frame=start_frame,
        end_frame=end_frame,
        reference_frame=reference_frame
    )

    return track


def _cast(individual_metadata: dict) -> Individual:
    individual_id = individual_metadata["individual_id"]
    tracks = [
        _cast_track(track_metadata)
        for track_metadata in individual_metadata["tracks"]
    ]

    individual = Individual(
        individual_id=individual_id,
        tracks=tracks
    )

    return individual


def get_individuals() -> list[Individual]:
    """
    Get the list of individuals from the metadata file.
    """
    individuals_metadata = _read()

    return [
        _cast(individual_metadata) for individual_metadata in individuals_metadata
    ]
