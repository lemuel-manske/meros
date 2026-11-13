from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from typing import Protocol

import numpy as np

"""
A bounding box is a rectangle that surrounds a subject in an image.
"""
type BBox = list[int]

"""
A spot is a tiny, rounded, dark dot on the Goliath Grouper head.
"""
type SubjectSpot = list[int]


class DomainObject:
    """
    Defines a DomainObject.

    It's a databag. May contain methods for logic.

    Only for heavy properties, use a _load_ fn property and expose as a cached_property.

    DomainObjects should aggregate their own reads.

    Always add python docs for DomainObjects.
    """

    ...


class Store(Protocol):
    """
    Defines a Store.

    It's a collection of DomainObjects. May contain methods for writes.

    Usually do not contain methods for reads, only for top-level reads.
    """

    ...


@dataclass(frozen=True)
class SubjectImage(DomainObject):
    """
    A subject image is a single image of a subject, along with its associated metadata.
    """

    subject_id: str
    image_id: int

    path: str

    capture_date: date

    _load_im: Callable[[], np.ndarray] = field(repr=False, compare=False)

    _load_crop: Callable[[], np.ndarray] = field(repr=False, compare=False)

    bbox: BBox = field(default_factory=list)
    spots: list[SubjectSpot] = field(default_factory=list)

    @cached_property
    def im(self) -> np.ndarray:
        return self._load_im()

    @cached_property
    def crop(self) -> np.ndarray:
        return self._load_crop()


@dataclass(frozen=True)
class Subject(DomainObject):
    """
    A subject is a single Goliath Grouper, along with its associated images.
    """

    subject_id: str

    images: list[SubjectImage] = field(default_factory=list)


class Subjects(Store, Protocol):
    """
    A store for Subjects and their associated images and metadata.
    """

    def get(self, subject_id: str) -> Subject: ...

    def get_all(self) -> list[Subject]: ...

    def write_subject(self, subject: Subject, source_video_id: str | None = None) -> None: ...

    def write_subject_crop(self, subject_id: str, image_id: int, crop: np.ndarray) -> None: ...

    def write_subject_spots(
        self, subject_id: str, image_id: int, spots: list[SubjectSpot]
    ) -> None: ...


@dataclass(frozen=True)
class FrameAnnotation(DomainObject):
    """
    A frame annotation is a single annotation of a video frame, along with its associated metadata.

    It's used to annootate the bounding boxes of Goliath Groupers in a video frame, as well as to indicate whether the frame has been reviewed or excluded from further processing.
    """

    recording_id: str

    suggestions: list[BBox] = field(default_factory=list)
    bboxes: list[BBox] = field(default_factory=list)

    """
    Reviewed indicates whether the frame has been reviewed by a human annotator.

    If True, the bboxes field contains the final bounding boxes for the frame.

    If False, the suggestions field contains the suggested bounding boxes for the frame (might include false positives).
    """
    reviewed: bool = False

    excluded: bool = False


@dataclass(frozen=True)
class FrameDetection(DomainObject):
    """A predicted Goliath Grouper bounding box and its confidence in a video frame."""

    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class VideoFrame(DomainObject):
    """
    A video frame is a single frame of a video, along with its associated metadata.
    """

    video_id: str
    frame_index: int

    capture_date: date

    path: str

    timestamp: float

    _load_im: Callable[[], np.ndarray] = field(repr=False, compare=False)

    annotations: FrameAnnotation | None = None

    detections: list[FrameDetection] = field(default_factory=list)

    @property
    def im(self) -> np.ndarray:
        return self._load_im()


@dataclass(frozen=True)
class Video(DomainObject):
    """
    A video is a single video, along with its associated frames.
    """

    video_id: str
    path: str

    """
    The frames selected by the finetuned model for further processing.
    """
    selected_frames: list[VideoFrame] = field(default_factory=list)

    frames: list[VideoFrame] = field(default_factory=list)


class Videos(Store, Protocol):
    """
    A store for Videos and their associated frames and metadata.
    """

    def get(self, video_id: str) -> Video: ...

    def get_all(self) -> list[Video]: ...

    def write_frame_annotation(self, frame: VideoFrame, annotation: FrameAnnotation) -> None: ...

    def write_video_detections(
        self, video: Video, detections: dict[int, list[FrameDetection]]
    ) -> None: ...

    def write_frame_neighbors(
        self, video: Video, pivot: VideoFrame, frames: list[VideoFrame]
    ) -> None: ...
