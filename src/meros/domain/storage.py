import numpy as np

from dataclasses import dataclass
from typing import (
    Literal,
    NamedTuple,
    Protocol,
)


class FrameMedia(NamedTuple):
    frame_idx: int
    data: np.ndarray


class MediaStore(Protocol):
    def has_frames(
        self,
        video_id: str,
    ) -> bool: ...

    def masked_crop_exists(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> bool: ...

    def aligned_crop_exists(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> bool: ...

    def composite_exists(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
    ) -> bool: ...

    def composite_enhanced_exists(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
    ) -> bool: ...

    def composite_comparison_exists(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
    ) -> bool: ...

    def composite_matches_exists(
        self,
        video_id: str,
        track_id: str,
        other_video_id: str,
        other_track_id: str,
        enhanced: bool = False,
    ) -> bool: ...

    def read_frame(
        self,
        video_id: str,
        frame_idx: int,
    ) -> np.ndarray: ...

    def read_frames(
        self,
        video_id: str,
    ) -> list[FrameMedia]: ...

    def write_frame(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
    ) -> None: ...

    def read_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> np.ndarray: ...

    def write_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        crop: np.ndarray,
    ) -> None: ...

    def read_masked_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> np.ndarray: ...

    def write_masked_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        masked_crop: np.ndarray,
    ) -> None: ...

    def remove_alignment(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> None: ...

    def read_aligned_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> np.ndarray: ...

    def write_aligned_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        aligned_crop: np.ndarray,
    ) -> None: ...

    def write_aligned_overlay(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        aligned_overlay: np.ndarray,
    ) -> None: ...

    def read_composite(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
    ) -> np.ndarray: ...

    def write_composite(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
        composite: np.ndarray,
    ) -> None: ...

    def read_composite_enhanced(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
    ) -> np.ndarray: ...

    def write_composite_enhanced(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
        composite_enhanced: np.ndarray,
    ) -> None: ...

    def read_composite_comparison(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
    ) -> np.ndarray: ...

    def write_composite_comparison(
        self,
        video_id: str,
        track_id: str,
        ref_frame_idx: int,
        composite_comparison: np.ndarray,
    ) -> None: ...

    def read_composite_matches(
        self,
        video_id: str,
        track_id: str,
        other_video_id: str,
        other_track_id: str,
        enhanced: bool = False,
    ) -> np.ndarray: ...

    def write_composite_matches(
        self,
        video_id: str,
        track_id: str,
        other_video_id: str,
        other_track_id: str,
        composite_matches: np.ndarray,
        enhanced: bool = False,
    ) -> None: ...

    def read_visualization(
        self,
        video_id: str,
        frame_idx: int,
    ) -> np.ndarray: ...

    def write_visualization(
        self,
        video_id: str,
        frame_idx: int,
        visualization: np.ndarray,
    ) -> None: ...


@dataclass(frozen=True)
class TrackObservation:
    bbox: list[int]
    mask_area: int


@dataclass(frozen=True)
class Track:
    initial_frame: int
    initial_bbox: list[int]
    frames: dict[str, TrackObservation]


@dataclass(frozen=True)
class TrackMetadata:
    video_id: str
    tracks: dict[str, Track]


@dataclass(frozen=True)
class TrackAnnotation:
    track_id: str
    start_frame: int
    end_frame: int
    viewpoint: str
    quality: int


AlignmentStatus = Literal[
    "insufficient_matches",
    "estimation_failed",
    "candidate",
    "rejected_geometry",
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


type BBox  = dict[str, list[int]]


type IndividualId = tuple[str, str, int, int]  # (video_id, track_id, start_frame, end_frame)

@dataclass(frozen=True)
class IndividualTrack:
    video_id: str
    track_id: str
    start_frame: int
    end_frame: int
    reference_frame: int

    @property
    def id(self) -> IndividualId:
        return (self.video_id, self.track_id, self.start_frame, self.end_frame)


@dataclass(frozen=True)
class Individual:
    individual_id: str
    tracks: list[IndividualTrack]


class MetadataStore(Protocol):
    def read_video(
        self,
        video_id: str,
    ) -> Video: ...

    def read_videos(
        self,
    ) -> list[Video]: ...

    def read_bboxes(
        self,
    ) -> dict[str, BBox]: ...

    def read_individuals(
        self,
    ) -> list[Individual]: ...

    def read_track(
        self,
        video_id: str,
    ) -> TrackMetadata: ...

    def write_track(
        self,
        video_id: str,
        metadata: TrackMetadata,
    ) -> None: ...

    def write_alignment(
        self,
        _id: IndividualId,
        reference_frame: int,
        rows: list[AlignmentMetadata],
    ) -> None: ...

    def read_alignment(
        self,
        _id: IndividualId,
    ) -> AlignmentRunMetadata: ...
