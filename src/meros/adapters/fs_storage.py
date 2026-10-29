import json
import os
import tempfile

import cv2 as cv
import numpy as np

from dataclasses import asdict, dataclass
from pathlib import Path

from meros.domain import (
    BBoxMap,
    EyeAnnotation,
    Subject,
    SubjectImage,
    Individual,
    Interval,
    Track,
    TrackMetadata,
    TrackObservation,
    Video,
    VideoAnnotation,
)
from meros.utils import read_image, write_image


MEDIA_ROOT = Path("data/media")
METADATA_ROOT = Path("data/metadata")


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

            stream.write("\n")

        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class MediaPaths:
    def subjects(self) -> Path:
        return MEDIA_ROOT / "subjects"

    def subject(self, subject_id: str) -> Path:
        return MEDIA_ROOT / "subjects" / subject_id

    def subject_image(self, subject_id: str, image_id: int) -> Path:
        return MEDIA_ROOT / "subjects" / subject_id / f"{image_id}.jpg"

    def frames(self, video_id: str) -> Path:
        return MEDIA_ROOT / "frames" / video_id

    def frame(self, video_id: str, frame_idx: int) -> Path:
        return MEDIA_ROOT / "frames" / video_id / f"{frame_idx}.jpg"

    def masked_crop(self, video_id: str, track_id: str, frame_idx: int) -> Path:
        return MEDIA_ROOT / "masked_crops" / video_id / track_id / f"{frame_idx}.png"


media_paths = MediaPaths()


class LocalFsMediaStore:
    paths: MediaPaths = media_paths

    def list_subjects(self) -> list[Subject]:
        root = self.paths.subjects()

        if not root.exists():
            return []

        return [
            Subject(directory.name, self.list_subject_images(directory.name))
            for directory in sorted(root.iterdir())
            if directory.is_dir()
        ]

    def list_subject_images(self, subject_id: str) -> list[SubjectImage]:
        directory = self.subject_image_path(subject_id, 0).parent

        if not directory.is_dir():
            return []

        records = [
            SubjectImage(subject_id, int(path.stem), str(path))
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg" and path.stem.isdecimal()
        ]
        return sorted(records, key=lambda item: item.image_id)

    def subject_image_path(self, subject_id: str, image_id: int) -> Path:
        if not subject_id or subject_id in {".", ".."} or "/" in subject_id or "\\" in subject_id:
            raise ValueError("Invalid subject ID")

        if image_id < 0:
            raise ValueError("Image ID must be nonnegative")

        return self.paths.subject_image(subject_id, image_id)

    def read_subject_image(self, subject_id: str, image_id: int) -> np.ndarray:
        return read_image(self.subject_image_path(subject_id, image_id))

    def frame_path(self, video_id: str, frame_idx: int) -> Path:
        return media_paths.frame(video_id, frame_idx)

    def frames_path(self, video_id: str) -> Path:
        return media_paths.frames(video_id)

    def frame_ids(self, video_id: str) -> list[int]:
        return sorted(int(p.stem) for p in self.frames_path(video_id).glob("*.jpg"))

    def read_frame(self, video_id: str, frame_idx: int) -> np.ndarray:
        return read_image(media_paths.frame(video_id, frame_idx))

    def write_frame(self, video_id: str, frame_idx: int, frame: np.ndarray) -> None:
        write_image(media_paths.frame(video_id, frame_idx), frame)

    def masked_crop_path(self, video_id: str, track_id: str, frame_idx: int) -> Path:
        return media_paths.masked_crop(video_id, track_id, frame_idx)

    def read_masked_crop(self, video_id: str, track_id: str, frame_idx: int) -> np.ndarray:
        return read_image(
            media_paths.masked_crop(video_id, track_id, frame_idx), cv.IMREAD_UNCHANGED
        )

    def write_masked_crop(
        self, video_id: str, track_id: str, frame_idx: int, masked_crop: np.ndarray
    ) -> None:
        write_image(media_paths.masked_crop(video_id, track_id, frame_idx), masked_crop)


@dataclass(frozen=True)
class MetadataPaths:
    def subject_eye(self, subject_id: str, image_id: int) -> Path:
        return METADATA_ROOT / "subjects" / subject_id / "eyes" / f"{image_id}.json"

    def subject_spots(self, subject_id: str, image_id: int) -> Path:
        return METADATA_ROOT / "subjects" / subject_id / "spots" / f"{image_id}.json"

    def bboxes(self) -> Path:
        return METADATA_ROOT / "videos" / "bboxes.json"

    def track(self, video_id: str) -> Path:
        return METADATA_ROOT / "tracks" / video_id / "metadata.json"

    def videos(self) -> Path:
        return METADATA_ROOT / "videos" / "videos.json"


metadata_paths = MetadataPaths()


class LocalFsMetadataStore:
    paths: MetadataPaths = metadata_paths

    def read_bboxes(self) -> dict[str, BBoxMap]:
        try:
            with metadata_paths.bboxes().open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            return {}

        if not isinstance(metadata, dict):
            raise ValueError("Seed boxes must be keyed by video and track")

        for tracks in metadata.values():
            if not isinstance(tracks, dict):
                raise ValueError("Seed boxes must be keyed by track")

        return metadata

    def write_bboxes(self, boxes: dict[str, BBoxMap]) -> None:
        write_json_atomic(metadata_paths.bboxes(), boxes)

    def read_video(self, video_id: str) -> Video:
        with metadata_paths.videos().open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        for video_metadata in metadata:
            if video_metadata.get("video_id") == video_id:
                return Video(
                    **{
                        **video_metadata,
                        "intervals": [
                            Interval(**interval) for interval in video_metadata.get("intervals", [])
                        ],
                        "annotations": [
                            VideoAnnotation(**annotation)
                            for annotation in video_metadata.get("annotations", [])
                        ],
                        "individuals": [
                            Individual(**individual)
                            for individual in video_metadata.get("individuals", [])
                        ],
                    }
                )

        raise ValueError(f"Video {video_id} not found")

    def read_videos(self) -> list[Video]:
        with metadata_paths.videos().open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return [
            Video(
                **{
                    **video_metadata,
                    "intervals": [
                        Interval(**interval) for interval in video_metadata.get("intervals", [])
                    ],
                    "annotations": [
                        VideoAnnotation(**annotation)
                        for annotation in video_metadata.get("annotations", [])
                    ],
                    "individuals": [
                        Individual(**individual)
                        for individual in video_metadata.get("individuals", [])
                    ],
                }
            )
            for video_metadata in metadata
        ]

    def read_track(self, video_id: str) -> TrackMetadata:
        path = metadata_paths.track(video_id)

        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict) or not isinstance(metadata.get("tracks"), dict):
            raise ValueError("Tracking metadata must contain a tracks object")

        for track in metadata["tracks"].values():
            if not isinstance(track, dict) or not isinstance(track.get("frames"), dict):
                raise ValueError("Track observations must be a frames object")

        tracks = {
            track_id: Track(
                initial_frame=track["initial_frame"],
                initial_bbox=track["initial_bbox"],
                frames={
                    frame_idx: TrackObservation(**observation)
                    for frame_idx, observation in track["frames"].items()
                },
            )
            for track_id, track in metadata["tracks"].items()
        }

        return TrackMetadata(
            video_id=metadata["video_id"],
            tracks=tracks,
            processed_frame_count=metadata["processed_frame_count"],
        )

    def write_track(self, video_id: str, metadata: TrackMetadata) -> None:
        path = metadata_paths.track(video_id)

        write_json_atomic(path, asdict(metadata))

    def read_subject_spots(self, subject_id: str, image_id: int) -> list[list[int]]:
        LocalFsMediaStore().subject_image_path(subject_id, image_id)
        path = self.paths.subject_spots(subject_id, image_id)

        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except FileNotFoundError:
            return []

        if (
            data.get("subject_id") != subject_id
            or data.get("image_id") != image_id
            or data.get("coordinate_system") != "subject_image_pixels"
        ):
            raise ValueError(f"Invalid subject spot annotation metadata: {path}")

        points = data.get("points")

        if not isinstance(points, list) or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(type(value) is not int or value < 0 for value in point)
            for point in points
        ):
            raise ValueError(f"Invalid spot coordinates: {path}")

        return points

    def write_subject_spots(self, subject_id: str, image_id: int, points: list[list[int]]) -> None:
        # Reuse media path validation rather than exposing filesystem details to processing.
        image_path = LocalFsMediaStore().subject_image_path(subject_id, image_id)
        if any(
            not isinstance(point, list)
            or len(point) != 2
            or any(type(value) is not int or value < 0 for value in point)
            for point in points
        ):
            raise ValueError("Invalid spot coordinates")

        write_json_atomic(
            self.paths.subject_spots(subject_id, image_id),
            {
                "subject_id": subject_id,
                "image_id": image_id,
                "image": str(image_path),
                "coordinate_system": "subject_image_pixels",
                "points": points,
            },
        )

    def read_subject_eye(self, subject_id: str, image_id: int) -> EyeAnnotation | None:
        LocalFsMediaStore().subject_image_path(subject_id, image_id)
        path = self.paths.subject_eye(subject_id, image_id)

        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except FileNotFoundError:
            return None

        if (
            not isinstance(payload, dict)
            or payload.get("subject_id") != subject_id
            or payload.get("image_id") != image_id
            or payload.get("side") != "left"
            or payload.get("coordinate_system") != "subject_image_pixels"
        ):
            raise ValueError(f"Invalid eye metadata: {path}")

        eye = payload.get("eye")

        if not isinstance(eye, dict):
            raise ValueError(f"Missing eye ellipse: {path}")

        try:
            center = eye["center"]
            radius_x = float(eye["radius_x"])
            radius_y = float(eye["radius_y"])
            angle = float(eye["angle_degrees"])

            if not (
                isinstance(center, list)
                and len(center) == 2
                and all(type(v) in (float, int) for v in center)
            ):
                raise ValueError("Invalid eye center")

            result = EyeAnnotation(
                center=(float(center[0]), float(center[1])),
                radius_x=radius_x,
                radius_y=radius_y,
                angle_degrees=angle,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid eye ellipse: {path}") from exc

        self._validate_eye(result)
        return result

    @staticmethod
    def _validate_eye(eye: EyeAnnotation) -> None:
        import math

        values = (*eye.center, eye.radius_x, eye.radius_y, eye.angle_degrees)

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Eye ellipse must have finite coordinates and radii")

        if eye.radius_x <= 0 or eye.radius_y <= 0:
            raise ValueError("Eye ellipse radii must be positive")

    def write_subject_eye(
        self, subject_id: str, image_id: int, eye: EyeAnnotation
    ) -> None:
        LocalFsMediaStore().subject_image_path(subject_id, image_id)
        self._validate_eye(eye)
        write_json_atomic(
            self.paths.subject_eye(subject_id, image_id),
            {
                "subject_id": subject_id,
                "image_id": image_id,
                "side": "left",
                "coordinate_system": "subject_image_pixels",
                "eye": {
                    "center": list(eye.center),
                    "radius_x": eye.radius_x,
                    "radius_y": eye.radius_y,
                    "angle_degrees": eye.angle_degrees,
                },
            },
        )
