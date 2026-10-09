from dataclasses import dataclass
from math import isfinite
from typing import Literal


def validate_bbox(bbox: list[int]) -> None:
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
    ):
        raise ValueError("Bounding boxes require four integer coordinates")

    x1, y1, x2, y2 = bbox

    if not 0 <= x1 <= x2 or not 0 <= y1 <= y2:
        raise ValueError("Bounding box coordinates must describe a nonempty region")


@dataclass(frozen=True)
class TrackObservation:
    bbox: list[int]
    mask_area: int

    def __post_init__(self) -> None:
        validate_bbox(self.bbox)

        x1, y1, x2, y2 = self.bbox

        if type(self.mask_area) is not int or not 0 < self.mask_area <= (x2 - x1 + 1) * (
            y2 - y1 + 1
        ):
            raise ValueError("Mask area must fit inside its bounding box")


@dataclass(frozen=True)
class Track:
    initial_frame: int
    initial_bbox: list[int]
    frames: dict[str, TrackObservation]

    def __post_init__(self) -> None:
        validate_bbox(self.initial_bbox)

        if type(self.initial_frame) is not int or self.initial_frame < 0:
            raise ValueError("Initial frame must be a nonnegative integer")


@dataclass(frozen=True)
class TrackMetadata:
    video_id: str
    tracks: dict[str, Track]
    processed_frame_count: int

    def __post_init__(self) -> None:
        if type(self.processed_frame_count) is not int or self.processed_frame_count <= 0:
            raise ValueError("Processed frame count must be a positive integer")


AlignmentStatus = Literal[
    "insufficient_matches", "estimation_failed", "accepted", "rejected_geometry"
]


@dataclass(frozen=True)
class AlignmentMetadata:
    status: AlignmentStatus
    matches: int
    inliers: int
    frame_idx: int | None = None
    inlier_ratio: float | None = None
    median_error_px: float | None = None
    inlier_hull_fraction: float | None = None
    source_to_reference: list[list[float]] | None = None

    def __post_init__(self) -> None:
        if self.status not in (
            "insufficient_matches",
            "estimation_failed",
            "accepted",
            "rejected_geometry",
        ):
            raise ValueError("Unknown alignment status")

        if (
            type(self.matches) is not int
            or type(self.inliers) is not int
            or not 0 <= self.inliers <= self.matches
        ):
            raise ValueError("Alignment counts must be consistent nonnegative integers")

        if self.frame_idx is not None and (type(self.frame_idx) is not int or self.frame_idx < 0):
            raise ValueError("Alignment frame must be a nonnegative integer")

        for value in (self.inlier_ratio, self.median_error_px, self.inlier_hull_fraction):
            if value is not None and (
                not isinstance(value, (int, float)) or not isfinite(value) or value < 0
            ):
                raise ValueError("Alignment statistics must be finite and nonnegative")

        if self.inlier_ratio is not None and self.inlier_ratio > 1:
            raise ValueError("Inlier ratio must be between zero and one")


@dataclass(frozen=True)
class AlignmentRunMetadata:
    video_id: str
    track_id: str
    start_frame: int
    end_frame: int
    reference_frame: int
    method: str
    frames: list[AlignmentMetadata]


@dataclass(frozen=True)
class Video:
    video_id: str
    mirrored: bool
    fname: str
    path: str
    frame_count: int | None = None

    def __post_init__(self) -> None:
        for value in (self.video_id, self.fname, self.path):
            if not isinstance(value, str) or not value:
                raise ValueError("Video identity and path must be nonempty strings")

        if type(self.mirrored) is not bool:
            raise ValueError("Video mirroring must be a boolean")

        if self.frame_count is not None and (
            type(self.frame_count) is not int or self.frame_count <= 0
        ):
            raise ValueError("Video frame count must be a positive integer")


type BBox = dict[str, list[int]]
type TrackSelectionKey = tuple[str, str, int, int, int]


@dataclass(frozen=True)
class TrackSelection:
    video_id: str
    track_id: str
    start_frame: int
    end_frame: int
    reference_frame: int

    @property
    def id(self) -> TrackSelectionKey:
        return (
            self.video_id,
            self.track_id,
            self.start_frame,
            self.end_frame,
            self.reference_frame,
        )

    @property
    def selection_id(self) -> str:
        return f"{self.video_id}--{self.track_id}--{self.start_frame}-{self.end_frame}--ref-{self.reference_frame}"

    def __post_init__(self) -> None:
        if not 0 <= self.start_frame <= self.reference_frame <= self.end_frame:
            raise ValueError("Invalid selection interval/reference")

        for value in (self.video_id, self.track_id):
            if not value or "--" in value or not all((c.isalnum() or c in "_-" for c in value)):
                raise ValueError("Selection IDs must use letters, digits, underscore or hyphen")


@dataclass(frozen=True)
class Individual:
    individual_id: str
    tracks: list[TrackSelection]
