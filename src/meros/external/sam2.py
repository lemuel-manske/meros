"""Translate the untyped SAM2 library into the application's predictor contract."""

import importlib

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, cast

from meros.domain.prediction import BinaryMask, Prediction, SeedBox


class BooleanTensor(Protocol):
    def cpu(self) -> "BooleanTensor": ...

    def numpy(self) -> BinaryMask: ...


class MaskLogits(Protocol):
    def gt(self, threshold: float) -> BooleanTensor: ...


class Sam2Backend(Protocol):
    def init_state(
        self,
        video_path: str,
        *,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
    ) -> object: ...

    def add_new_points_or_box(
        self,
        *,
        inference_state: object,
        frame_idx: int,
        obj_id: int,
        box: SeedBox,
    ) -> object: ...

    def propagate_in_video(
        self, state: object
    ) -> Iterator[tuple[int, Sequence[int], Sequence[MaskLogits]]]: ...


class Sam2Builder(Protocol):
    def __call__(
        self,
        config_file: str,
        ckpt_path: str,
        *,
        device: str,
    ) -> Sam2Backend: ...


class SAM2Predictor:
    def __init__(self, backend: Sam2Backend) -> None:
        self.backend = backend

    def init_state(self, video_path: str) -> object:
        return self.backend.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )

    def add_new_points_or_box(
        self,
        inference_state: object,
        frame_idx: int,
        obj_id: int,
        box: SeedBox,
    ) -> None:
        self.backend.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )

    def propagate_in_video(self, state: object) -> Iterator[Prediction]:
        # These SAM2 methods are already decorated with torch.inference_mode.
        for frame_idx, object_ids, logits in self.backend.propagate_in_video(state):
            masks: dict[str, BinaryMask] = {}

            for object_id, logits_mask in zip(object_ids, logits, strict=True):
                mask = logits_mask.gt(0).cpu().numpy()

                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]

                if mask.ndim != 2:
                    raise ValueError("SAM2 must return one two-dimensional mask per object")

                masks[str(int(object_id))] = mask

            yield Prediction(frame_idx, masks)


def build_sam2_predictor(
    checkpoint: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt"),
) -> SAM2Predictor:
    module = importlib.import_module("sam2.build_sam")

    # SAM2 does not declare this public interface. The adapter is the sole
    # typed boundary; processing never handles Torch tensors or SAM2 state.
    builder = cast(Sam2Builder, module.build_sam2_video_predictor)

    backend = builder(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(checkpoint),
        device="cuda",
    )

    return SAM2Predictor(backend)
