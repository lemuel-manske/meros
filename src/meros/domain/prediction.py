from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

"""
A binary mask is a 2D array of booleans, where True indicates the presence of a subject and False indicates the background.
"""
type BinaryMask = NDArray[np.bool_]

"""
A seed box is a 4D array of floats, where the first two dimensions are the coordinates of the top-left and bottom-right corners of the box, and the last two dimensions are the width and height of the box.
"""
type SeedBox = NDArray[np.float32]


@dataclass(frozen=True)
class Prediction:
    """
    A prediction is a single prediction of a subject in an image, along with its associated metadata.
    """

    frame_idx: int
    masks: dict[str, BinaryMask]


class VideoPredictor(Protocol):
    """
    Defines a VideoPredictor.
    """

    def init_state(self, video_path: str) -> object: ...

    def add_new_points_or_box(
        self,
        inference_state: object,
        frame_idx: int,
        obj_id: int,
        box: SeedBox,
    ) -> None: ...

    def propagate_in_video(self, state: object) -> Iterator[Prediction]: ...
