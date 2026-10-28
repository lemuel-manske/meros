import json
import os
import tempfile

import cv2 as cv
import numpy as np

from dataclasses import asdict, dataclass
from pathlib import Path

from meros.domain import (
    BBoxMap,
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
    def frames(self, video_id: str) -> Path:
        return MEDIA_ROOT / "frames" / video_id

    def frame(self, video_id: str, frame_idx: int) -> Path:
        return MEDIA_ROOT / "frames" / video_id / f"{frame_idx}.jpg"

    def masked_crop(self, video_id: str, track_id: str, frame_idx: int) -> Path:
        return MEDIA_ROOT / "masked_crops" / video_id / track_id / f"{frame_idx}.png"


media_paths = MediaPaths()


class LocalFsMediaStore:
    paths: MediaPaths = media_paths

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
    def bboxes(self) -> Path:
        return METADATA_ROOT / "videos" / "bboxes.json"

    def track(self, video_id: str) -> Path:
        return METADATA_ROOT / "tracks" / video_id / "metadata.json"

    def videos(self) -> Path:
        return METADATA_ROOT / "videos" / "videos.json"

    def spots(self, video_id: str, track_id: str, frame_idx: int) -> Path:
        return METADATA_ROOT / "spots" / video_id / track_id / f"{frame_idx}.json"


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

    def read_spots(self, video_id: str, track_id: str, frame_idx: int) -> list[list[int]]:
        path = metadata_paths.spots(video_id, track_id, frame_idx)

        try:
            with path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        except FileNotFoundError:
            return []

        if (
            metadata.get("video_id") != video_id
            or metadata.get("track_id") != track_id
            or metadata.get("frame") != frame_idx
            or metadata.get("coordinate_system") != "masked_crop_pixels"
        ):
            raise ValueError(f"Invalid spot annotation metadata: {path}")

        points = metadata.get("points")

        if not isinstance(points, list) or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(type(coordinate) is not int or coordinate < 0 for coordinate in point)
            for point in points
        ):
            raise ValueError(f"Invalid spot coordinates: {path}")

        return points

    def write_spots(
        self, video_id: str, track_id: str, frame_idx: int, points: list[list[int]]
    ) -> None:
        path = metadata_paths.spots(video_id, track_id, frame_idx)
        image = media_paths.masked_crop(video_id, track_id, frame_idx)

        write_json_atomic(
            path,
            {
                "video_id": video_id,
                "track_id": track_id,
                "frame": frame_idx,
                "image": str(image),
                "coordinate_system": "masked_crop_pixels",
                "points": points,
            },
        )
