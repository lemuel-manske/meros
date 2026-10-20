"""Typed CPU-facing predictions; inference state stays owned by the predictor."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


type BinaryMask = NDArray[np.bool_]
type SeedBox = NDArray[np.float32]


@dataclass(frozen=True)
class Prediction:
    frame_idx: int
    masks: dict[str, BinaryMask]


class VideoPredictor(Protocol):
    def init_state(self, video_path: str) -> object: ...

    def add_new_points_or_box(
        self,
        inference_state: object,
        frame_idx: int,
        obj_id: int,
        box: SeedBox,
    ) -> None: ...

    def propagate_in_video(self, state: object) -> Iterator[Prediction]: ...
