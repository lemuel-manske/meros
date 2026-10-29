from dataclasses import dataclass, field
from typing import Literal


type BBox = list[int]
type BBoxMap = dict[str, BBox]

type Viewpoint = Literal["frontal", "left", "right", "back"]

type TrackSelectionKey = tuple[str, str, int, int]  # (video_id, track_id, start_frame, end_frame)


@dataclass(frozen=True)
class TrackObservation:
    bbox: BBox
    mask_area: int


@dataclass(frozen=True)
class Track:
    initial_frame: int
    initial_bbox: BBox
    frames: dict[str, TrackObservation]


@dataclass(frozen=True)
class TrackMetadata:  # bound to metadata.json
    video_id: str
    tracks: dict[str, Track]
    processed_frame_count: int


@dataclass(frozen=True)
class TrackSelection:
    video_id: str
    track_id: str
    start_frame: int
    end_frame: int

    @property
    def id(self) -> TrackSelectionKey:
        return (
            self.video_id,
            self.track_id,
            self.start_frame,
            self.end_frame,
        )

    @property
    def selection_id(self) -> str:
        return f"{self.video_id}--{self.track_id}"


@dataclass(frozen=True)
class Individual:
    individual_id: str
    track_id: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class VideoAnnotation:
    track_id: str
    start_frame: int
    end_frame: int
    viewpoint: Viewpoint
    quality: int


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


@dataclass(frozen=True)
class Video:
    video_id: str
    fname: str
    path: str

    mirrored: bool = False

    intervals: list[Interval] = field(default_factory=list)
    annotations: list[VideoAnnotation] = field(default_factory=list)
    individuals: list[Individual] = field(default_factory=list)

    def matching_annotations(
        self,
        *,
        viewpoint: Viewpoint | None = None,
        min_quality: int | None = None,
        track_id: str | None = None,
    ) -> list[VideoAnnotation]:
        return [
            annotation
            for annotation in self.annotations
            if (viewpoint is None or annotation.viewpoint == viewpoint)
            and (min_quality is None or annotation.quality >= min_quality)
            and (track_id is None or annotation.track_id == track_id)
        ]


@dataclass(frozen=True)
class SubjectImage:
    subject_id: str
    image_id: int
    path: str


@dataclass(frozen=True)
class Subject:
    subject_id: str
    images: list[SubjectImage] = field(default_factory=list)


@dataclass(frozen=True)
class EyeAnnotation:
    """Visible left-eye outline in original subject-image pixels."""

    center: tuple[float, float]
    radius_x: float
    radius_y: float
    angle_degrees: float = 0.0
