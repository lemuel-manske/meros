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


@dataclass
class SubjectImage(DomainObject):
    """
    A subject image is a single image of a subject, along with its associated metadata.
    """

    _load_im: Callable[[], np.ndarray] = field(repr=False, compare=False)
    _load_binary_im: Callable[[], np.ndarray | None] = field(repr=False, compare=False)
    _load_crop_im: Callable[[], np.ndarray] = field(repr=False, compare=False)

    _id: int

    path: str

    capture_date: date

    bbox: BBox = field(default_factory=list)

    spots: list[SubjectSpot] = field(default_factory=list)

    @cached_property
    def im(self) -> np.ndarray:
        return self._load_im()

    @cached_property
    def binary_im(self) -> np.ndarray | None:
        return self._load_binary_im()

    @cached_property
    def crop_im(self) -> np.ndarray:
        return self._load_crop_im()


@dataclass
class Subject(DomainObject):
    """
    A subject is a single Goliath Grouper, along with its associated images.
    """

    _id: str

    images: list[SubjectImage] = field(default_factory=list)

    def get_image(self, image_id: int) -> SubjectImage | None:
        for image in self.images:
            if image._id == image_id:
                return image

        return None


class Subjects(Store, Protocol):
    """
    A store for Subjects and their associated images and metadata.
    """

    def get(self, subject_id: str) -> Subject: ...

    def get_all(self) -> list[Subject]: ...

    def write(self, subject: Subject) -> None: ...


@dataclass
class VideoFrame(DomainObject):
    """
    A video frame is a single frame of a video, along with its associated metadata.
    """

    _load_im: Callable[[], np.ndarray] = field(repr=False, compare=False)

    _id: int

    path: str

    capture_date: date

    bboxes: list[BBox] = field(default_factory=list)
    reviewed: bool = False
    excluded: bool = False

    @property
    def im(self) -> np.ndarray:
        return self._load_im()


@dataclass
class Video(DomainObject):
    """
    A video is a single video, along with its associated frames.
    """

    _id: str

    path: str

    frames: list[VideoFrame] = field(default_factory=list)


class Videos(Store, Protocol):
    """
    A store for Videos and their associated frames and metadata.
    """

    def get(self, video_id: str) -> Video: ...

    def get_all(self) -> list[Video]: ...

    def write(self, video: Video) -> None: ...
