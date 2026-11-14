import csv
from dataclasses import asdict
from datetime import date
from functools import partial
from pathlib import Path

import cv2 as cv
import numpy as np

from meros.adapters.paths import media_paths, metadata_paths
from meros.domain.storage import (
    FrameAnnotation,
    FrameDetection,
    Video,
    VideoFrame,
    Videos,
)
from meros.utils import is_image_file, is_video_file, read_image, read_json, write_json


def read_videos() -> list[Path]:
    root = media_paths.videos()

    return sorted(path for path in root.iterdir() if is_video_file(path))


def read_video(path: Path) -> Video:
    return Video(
        _id=path.stem,
        path=str(path),
        selected_frames=read_selected_frames(path),
        frames=read_video_frames(path),
    )


def read_video_capture_date(video_id: str) -> date:
    path = metadata_paths.video_annotations()

    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["id"] == video_id:
                return date.fromisoformat(row["capture_date"])

    assert False, f"Video {video_id} not found in {path}"


def read_selected_frames(path: Path) -> list[VideoFrame]:
    directory = media_paths.selected_video_frames(path.stem)

    if not directory.exists():
        return []

    fps = cv.VideoCapture(str(path)).get(cv.CAP_PROP_FPS)

    return [
        VideoFrame(
            capture_date=read_video_capture_date(path.stem),
            path=str(frame_path),
            _id=int(frame_path.stem),
            _load_im=partial(read_image, frame_path),
            annotations=None,
            detections=[],
        )
        for frame_path in sorted(directory.iterdir())
        if is_image_file(frame_path)
    ]


def read_video_frames(path: Path) -> list[VideoFrame]:
    capture = cv.VideoCapture(str(path), cv.CAP_FFMPEG, [cv.CAP_PROP_N_THREADS, 2])

    try:
        assert capture.isOpened(), f"Could not open video: {path}"

        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

        assert frame_count >= 0
    finally:
        capture.release()

    annotations = read_json(metadata_paths.video_detections(path.stem), default={})
    predictions = read_json(metadata_paths.video_predictions(path.stem), default={})

    return [
        VideoFrame(
            capture_date=read_video_capture_date(path.stem),
            path=str(path),
            _id=index,
            _load_im=partial(read_video_frame, path, index),
            annotations=(
                FrameAnnotation(**annotations[str(index)]) if str(index) in annotations else None
            ),
            detections=[FrameDetection(**item) for item in predictions.get(str(index), [])],
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

    def write_frame_annotation(
        self, video_id: str, frame: VideoFrame, annotation: FrameAnnotation
    ) -> None:
        path = metadata_paths.video_detections(video_id)

        annotations = read_json(path, default={})
        annotations[str(frame._id)] = asdict(annotation)

        write_json(path, annotations)

    def write_video_detections(
        self, video: Video, detections: dict[int, list[FrameDetection]]
    ) -> None:
        path = metadata_paths.video_predictions(video._id)

        write_json(
            path,
            {str(index): [asdict(item) for item in items] for index, items in detections.items()},
        )
