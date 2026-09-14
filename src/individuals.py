from typing import TypedDict


class TrackSelection(TypedDict):
    video_id: str
    track_id: str
    start_frame: int
    end_frame: int
    reference_frame: int


class IndividualSelection(TypedDict):
    individual_id: str
    tracks: list[TrackSelection]

# keep track of known individuals and their associated tracks,
# along with the start and end frames for each track
INDIVIDUALS: list[IndividualSelection] = [
    {
        "individual_id": "4",
        "tracks": [
            {
                "video_id": "65_5",
                "track_id": "1",
                "start_frame": 55,
                "end_frame": 63,
                "reference_frame": 59,
            },
            {
                "video_id": "65_6",
                "track_id": "1",
                "start_frame": 82,
                "end_frame": 98,
                "reference_frame": 90,
            },
        ],
    },
]
