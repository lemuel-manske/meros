from collections.abc import Sequence
import importlib
from pathlib import Path
import sys
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from meros.domain import BinaryMask

IMAGE_SIZE = (1024, 1024)
EPOCHS = 1000
LEARNING_RATE = 0.001
THRESHOLD = 0.5


class MaskTensor(Protocol):
    def squeeze(self) -> "MaskTensor": ...

    def detach(self) -> "MaskTensor": ...

    def cpu(self) -> "MaskTensor": ...

    def numpy(self) -> NDArray[np.float32]: ...


class GloPERBackend(Protocol):
    def __call__(
        self,
        image_o: Image.Image,
        *,
        pxl_size: tuple[int, int],
        kernal_filters: list[int],
        epochs: int,
        learning_rate: float,
        num_classes: int,
        t: float,
        reverse: bool,
        optimizer_name: str,
    ) -> Sequence[MaskTensor]: ...


class GloPERPredictor:
    def __init__(self, backend: GloPERBackend) -> None:
        self.backend = backend

    def predict(
        self,
        image: Image.Image,
        kernels: list[int],
        *,
        light_patterns: bool = True,
    ) -> dict[int, BinaryMask]:
        tensors = self.backend(
            image_o=image.convert("RGB"),
            pxl_size=IMAGE_SIZE,
            kernal_filters=kernels,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            num_classes=2,
            t=THRESHOLD,
            reverse=False,
            optimizer_name="AdamW",
        )

        masks: dict[int, BinaryMask] = {}

        for kernel_size, tensor in zip(kernels, tensors, strict=True):
            mask = tensor.squeeze().detach().cpu().numpy() > 0.5

            assert mask.ndim == 2

            if light_patterns:
                mask = ~mask

            masks[kernel_size] = mask

        return masks


def build_gloper_predictor(
    *,
    gloper_src: Path = Path("external/gloper"),
    device: str = "cuda",
) -> GloPERPredictor:
    gloper_src = gloper_src.resolve()

    if str(gloper_src) not in sys.path:
        sys.path.insert(0, str(gloper_src))

    module = importlib.import_module("gloper")

    setattr(module, "device", getattr(module.torch, "device")(device))

    backend = cast(GloPERBackend, module.binary_mask_gloper)

    return GloPERPredictor(backend)
