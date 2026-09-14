import torch

from sam2.build_sam import build_sam2_video_predictor

from src.consts import (
    SAM2_CHECKPOINT,
    SAM2_MODEL_CONFIG,
)


def build_sam2_predictor():
    return build_sam2_video_predictor(
        SAM2_MODEL_CONFIG,
        SAM2_CHECKPOINT,
        device="cuda",  # use GPU for faster inference
        dtype=torch.bfloat16,  # pyright: ignore
    )
