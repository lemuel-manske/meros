import csv
from dataclasses import asdict
from datetime import date
from functools import partial
from pathlib import Path

import cv2 as cv
import numpy as np

from meros.domain.storage import (
    BBox,
    FrameAnnotation,
    FrameDetection,
    Subject,
    SubjectImage,
    Subjects,
    SubjectSpot,
    Video,
    VideoFrame,
    Videos,
)
from meros.utils import is_video_file, read_image, read_json, write_image, write_json

"""
Always extract constants for paths and directories.
"""

MEDIA_ROOT = Path("data/media")
METADATA_ROOT = Path("data/metadata")

MEDIA_SUBJECTS_ROOT = MEDIA_ROOT / "subjects"
VIDEOS_ROOT = MEDIA_ROOT / "videos"
SELECTED_VIDEO_FRAMES_ROOT = VIDEOS_ROOT / "selected"
VIDEO_FRAME_NEIGHBORS_ROOT = VIDEOS_ROOT / "neighbors"

METADATA_SUBJECTS_ROOT = METADATA_ROOT / "subjects"
VIDEOS_METADATA_ROOT = METADATA_ROOT / "videos"

VIDEO_DETECTIONS_ROOT = VIDEOS_METADATA_ROOT / "detections"
VIDEO_PREDICTIONS_ROOT = VIDEOS_METADATA_ROOT / "predictions"


class MediaPaths:
    def videos(self) -> Path:
        return VIDEOS_ROOT

    def selected_video_frames(self, video_id: str) -> Path:
        return SELECTED_VIDEO_FRAMES_ROOT / video_id

    def frame_neighbors(self, video_id: str, pivot_index: int) -> Path:
        return VIDEO_FRAME_NEIGHBORS_ROOT / video_id / f"{pivot_index:08d}"

    def subjects(self) -> Path:
        return MEDIA_SUBJECTS_ROOT

    def subject(self, subject_id: str) -> Path:
        return MEDIA_SUBJECTS_ROOT / subject_id

    def subject_image(self, subject_id: str, image_id: int) -> Path:
        return MEDIA_SUBJECTS_ROOT / subject_id / f"{image_id}.jpg"

    def subject_crop(self, subject_id: str, image_id: int) -> Path:
        return self.subject(subject_id) / "crops" / f"{image_id}.png"

    def subject_bodies(self, subject_id: str) -> Path:
        return MEDIA_SUBJECTS_ROOT / subject_id / "body"

    def subject_body(self, subject_id: str, image_id: int) -> Path:
        return MEDIA_SUBJECTS_ROOT / subject_id / "body" / f"{image_id}.png"


class MetadataPaths:
    def video_predictions(self, video_id: str) -> Path:
        return VIDEO_PREDICTIONS_ROOT / f"{video_id}.json"

    def video_detections(self, video_id: str) -> Path:
        return VIDEO_DETECTIONS_ROOT / f"{video_id}.json"

    def video_annotations(self) -> Path:
        return VIDEOS_METADATA_ROOT / "annotations.csv"

    def subject_annotations(self, subject_id: str) -> Path:
        return METADATA_SUBJECTS_ROOT / subject_id / "annotations.csv"

    def subject_segmentation(self, subject_id: str) -> Path:
        return METADATA_SUBJECTS_ROOT / subject_id / "segmentation.json"

    def subject_spot_correspondences(self, subject_id: str) -> Path:
        return METADATA_SUBJECTS_ROOT / subject_id / "spot_correspondences"

    def subject_spots(self, subject_id: str, image_id: int) -> Path:
        return METADATA_SUBJECTS_ROOT / subject_id / "spots" / f"{image_id}.json"


media_paths = MediaPaths()
metadata_paths = MetadataPaths()


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
            timestamp=float(frame_path.stem) / fps,
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

        fps = capture.get(cv.CAP_PROP_FPS)

        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

        assert fps > 0 and frame_count >= 0
    finally:
        capture.release()

    annotations = read_json(metadata_paths.video_detections(path.stem), default={})
    predictions = read_json(metadata_paths.video_predictions(path.stem), default={})

    return [
        VideoFrame(
            capture_date=read_video_capture_date(path.stem),
            path=str(path),
            _id=index,
            timestamp=index / fps,
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


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


def read_subject_bbox(subject_id: str, image_id: int) -> BBox:
    path = metadata_paths.subject_segmentation(subject_id)

    bboxes = read_json(path, default={})

    return bboxes.get("bboxes", {}).get(str(image_id), [])


def read_subject_spots(subject_id: str, image_id: int) -> list[SubjectSpot]:
    path = metadata_paths.subject_spots(subject_id, image_id)

    data = read_json(path, default={"points": []})

    return data["points"]


def read_subject_crop(subject_id: str, image_id: int) -> np.ndarray:
    return read_image(media_paths.subject_crop(subject_id, image_id), cv.IMREAD_UNCHANGED)


def read_body(subject_id: str, image_id: int) -> np.ndarray | None:
    p = media_paths.subject_body(subject_id, image_id)

    if not p.exists():
        return None

    return read_image(p, cv.IMREAD_UNCHANGED)


def read_subject_capture_date(subject_id: str, frame_idx: int) -> date:
    path = metadata_paths.subject_annotations(subject_id)

    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["id"]) == frame_idx:
                return date.fromisoformat(row["capture_date"])

    assert False, f"Frame {frame_idx} not found in {path}"


def read_subject_images(subject_id: str) -> list[SubjectImage]:
    directory = media_paths.subject(subject_id)

    if not directory.exists():
        return []

    records = [
        SubjectImage(
            subject_id=subject_id,
            image_id=int(path.stem),
            capture_date=read_subject_capture_date(subject_id, int(path.stem)),
            path=str(path),
            _load_im=partial(read_image, path),
            _load_crop_im=partial(read_subject_crop, subject_id, int(path.stem)),
            _load_binary_im=partial(read_body, subject_id, int(path.stem)),
            bbox=read_subject_bbox(subject_id, int(path.stem)),
            spots=read_subject_spots(subject_id, int(path.stem)),
        )
        for path in directory.iterdir()
        if is_image_file(path)
    ]

    return sorted(records, key=lambda item: item.image_id)


class SubjectsAdapter(Subjects):
    def get_all(self) -> list[Subject]:
        root = media_paths.subjects()

        if not root.exists():
            return []

        return [
            Subject(
                subject_id=directory.name,
                images=read_subject_images(directory.name),
            )
            for directory in sorted(root.iterdir())
            if directory.is_dir()
        ]

    def get(self, subject_id: str) -> Subject:
        for subject in self.get_all():
            if subject.subject_id == subject_id:
                return subject

        raise ValueError(f"Subject {subject_id} not found")

    def write_body(self, im: np.ndarray, subject_id: str, image_id: int) -> None:
        path = media_paths.subject_body(subject_id, image_id)

        path.parent.mkdir(parents=True, exist_ok=True)

        write_image(path, im)

    def write_subject(self, subject: Subject, source_video_id: str | None = None) -> None:
        directory = media_paths.subject(subject.subject_id)

        directory.mkdir(parents=True, exist_ok=True)

        for image in subject.images:
            write_image(media_paths.subject_image(subject.subject_id, image.image_id), image.im)

            if image.bbox:
                path = metadata_paths.subject_segmentation(subject.subject_id)
                bboxes = read_json(path, default={})

                bboxes.setdefault("bboxes", {})[str(image.image_id)] = image.bbox

                write_json(path, bboxes)

            if image.spots:
                self.write_subject_spots(subject.subject_id, image.image_id, image.spots)

            if image.capture_date:
                path = metadata_paths.subject_annotations(subject.subject_id)

                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)

                    with path.open("w", newline="") as stream:
                        writer = csv.DictWriter(
                            stream, fieldnames=["id", "capture_date", "source_video_id"]
                        )
                        writer.writeheader()

                with path.open("a", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=["id", "capture_date", "source_video_id"]
                    )
                    writer.writerow(
                        {
                            "id": image.image_id,
                            "capture_date": image.capture_date.isoformat(),
                            "source_video_id": source_video_id,
                        }
                    )

    def write_subject_spots(self, subject_id: str, image_id: int, spots: list[SubjectSpot]) -> None:
        path = metadata_paths.subject_spots(subject_id, image_id)

        path.parent.mkdir(parents=True, exist_ok=True)

        write_json(path, {"points": spots})

    def write_subject_bbox(self, subject_id: str, image_id: int, bbox: BBox) -> None:
        path = metadata_paths.subject_segmentation(subject_id)
        data = read_json(path, default={})
        data.setdefault("bboxes", {})[str(image_id)] = bbox
        write_json(path, data)

    def write_subject_crop(self, subject_id: str, image_id: int, crop: np.ndarray) -> None:
        write_image(media_paths.subject_crop(subject_id, image_id), crop)


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

    def write_frame_neighbors(
        self, video: Video, pivot: VideoFrame, frames: list[VideoFrame]
    ) -> None:
        directory = media_paths.frame_neighbors(video._id, pivot._id)
        pending = {
            frame._id: directory / f"{frame._id:08d}.jpg"
            for frame in frames
            if not (directory / f"{frame._id:08d}.jpg").is_file()
        }

        if not pending:
            return

        capture = cv.VideoCapture(video.path, cv.CAP_FFMPEG, [cv.CAP_PROP_N_THREADS, 2])

        try:
            assert capture.isOpened(), f"Could not open video: {video.path}"

            capture.set(cv.CAP_PROP_POS_FRAMES, min(pending))

            for index in range(min(pending), max(pending) + 1):
                ok = capture.grab()
                assert ok, f"Could not decode {video.path}, frame {index}"

                if index not in pending:
                    continue

                ok, image = capture.retrieve()
                assert ok, f"Could not decode {video.path}, frame {index}"

                write_image(pending[index], image)
        finally:
            capture.release()
