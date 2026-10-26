"""Utilities shared by research experiments."""

import os
import uuid

import matplotlib.pyplot as plt
import numpy as np

from cv2.typing import MatLike
from numpy.typing import NDArray

from datetime import datetime, timezone
from pathlib import Path


DEBUG: bool = os.getenv("DEBUG", "0") == "1"


type Image = NDArray[np.uint8]
type Points = NDArray[np.float32]


def is_debug() -> bool:
    return DEBUG


def debug_img(img: Image | MatLike) -> None:
    if not is_debug():
        return
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.show()


def experiment_output(experiment_id: str) -> Path:
    return (
        Path("results")
        / experiment_id
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    )
