import os
from pathlib import Path

from ultralytics.engine.model import Model


def YOLO(model: str | Path) -> Model:
    if isinstance(model, str):
        model = Path(model)

    assert model.is_file(), f"YOLO model file does not exist: {model}"

    config_dir = model.parent / ".config"

    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    from ultralytics import YOLO as _YOLO

    return _YOLO(str(model))
