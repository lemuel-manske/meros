import csv
from datetime import date
from functools import partial
from pathlib import Path

import cv2 as cv
import numpy as np

from meros.adapters.paths import media_paths, metadata_paths
from meros.domain.storage import (
    Video,
    VideoFrame,
    Videos,
)
from meros.utils import is_video_file


def read_videos() -> list[Path]:
    root = media_paths.videos()

    return sorted(path for path in root.iterdir() if is_video_file(path))


def read_video(path: Path) -> Video:
    return Video(
        _id=path.stem,
        path=str(path),
        frames=read_video_frames(path),
    )


def read_video_capture_date(video_id: str) -> date:
    path = metadata_paths.video_annotations()

    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["id"] == video_id:
                return date.fromisoformat(row["capture_date"])

    assert False, f"Video {video_id} not found in {path}"


def read_video_frames(path: Path) -> list[VideoFrame]:
    capture = cv.VideoCapture(str(path), cv.CAP_FFMPEG, [cv.CAP_PROP_N_THREADS, 2])

    try:
        assert capture.isOpened(), f"Could not open video: {path}"

        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

        assert frame_count >= 0
    finally:
        capture.release()

    return [
        VideoFrame(
            capture_date=read_video_capture_date(path.stem),
            path=str(path),
            _id=index,
            _load_im=partial(read_video_frame, path, index),
        )
        for index in range(frame_count)
    ]


def read_video_frame(path: Path, index: int) -> np.ndarray:
    capture = cv.VideoCapture(str(path), cv.CAP_FFMPEG, [cv.CAP_PROP_N_THREADS, 2])

    try:
        assert capture.isOpened(), f"Could not open video: {path}"

        capture.set(cv.CAP_PROP_POS_FRAMES, index)

        ok, image = capture.read()
        assert ok, f"Could not decode {path}, frame {index}"

        return image
    finally:
        capture.release()


class VideosAdapter(Videos):
    def get_all(self) -> list[Video]:
        return [read_video(path) for path in read_videos()]

    def get(self, video_id: str) -> Video:
        for path in read_videos():
            if path.stem == video_id:
                return read_video(path)

        raise ValueError(f"Video {video_id} not found")

    def write(self, video: Video) -> None: ...
