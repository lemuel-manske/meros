import json
import os
from pathlib import Path
import tempfile
from typing import TypeVar, cast

T = TypeVar("T")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

            stream.write("\n")

        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path: Path, default: T) -> T:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return cast(T, json.load(stream))
    except FileNotFoundError:
        return default
