import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

from src.consts import (
    SAM2_CHECKPOINT,
    SAM2_MODEL_CONFIG,
)


_DEVICE = "cuda"  # use GPU for faster inference

_OFFLOAD_STATE_TO_CPU = False  # keep state on GPU for faster inference
_OFFLOAD_VIDEO_TO_CPU = True  # use CPU memory for video frames to reduce GPU memory usage


class SAM2Predictor():
    """
    Wraps SAM2 predictor to provide a consistent interface for video propagation.
    """

    def __init__(self, sam2_predictor):
        self.sam2_predictor = sam2_predictor

    def init_state(self, video_path: str) -> dict[str, torch.Tensor]:
        """
        Initialize the inference state for video propagation.
        """
        return self.sam2_predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=_OFFLOAD_VIDEO_TO_CPU,
            offload_state_to_cpu=_OFFLOAD_STATE_TO_CPU,
        )

    def propagate_in_video(self, state: dict[str, torch.Tensor]):
        """
        Propagate the masks in the video using the initialized state.
        """
        return self.sam2_predictor.propagate_in_video(state)

    def add_new_points_or_box(
        self,
        inference_state: dict[str, torch.Tensor],
        frame_idx: int,
        obj_id: int,
        box: np.ndarray,
    ) -> None:
        """
        Add new points or a bounding box to the inference state for a specific object.
        """
        self.sam2_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )


def build_sam2_predictor() -> SAM2Predictor:
    """
    Build a SAM2 predictor for video propagation.
    """
    sam2_predictor = build_sam2_video_predictor(
        SAM2_MODEL_CONFIG,
        SAM2_CHECKPOINT,
        device=_DEVICE,
        dtype=torch.bfloat16,  # pyright: ignore
    )

    return SAM2Predictor(sam2_predictor)
