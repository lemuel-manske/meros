import json
import hashlib

import cv2 as cv
import numpy as np

from dataclasses import asdict, dataclass
from pathlib import Path

from meros.domain import (
    BBoxMap,
    EyeAnnotation,
    SpotRegionReview,
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
from meros.utils import read_image, read_json, write_image, write_json_atomic


MEDIA_ROOT = Path("data/media")
METADATA_ROOT = Path("data/metadata")

MEDIA_SUBJECTS_ROOT = MEDIA_ROOT / "subjects"
FRAMES_ROOT = MEDIA_ROOT / "frames"
MASKED_CROPS_ROOT = MEDIA_ROOT / "masked_crops"

METADATA_SUBJECTS_ROOT = METADATA_ROOT / "subjects"
VIDEOS_METADATA_ROOT = METADATA_ROOT / "videos"
TRACKS_METADATA_ROOT = METADATA_ROOT / "tracks"

SPOT_REGION_REVIEWS_DIR = "spot_region_reviews"
SPOT_CORRESPONDENCES_DIR = "spot_correspondences"
MOUTHS_DIR = "mouths"
EYES_DIR = "eyes"
SPOTS_DIR = "spots"

BBOXES_PATH = VIDEOS_METADATA_ROOT / "bboxes.json"
VIDEOS_PATH = VIDEOS_METADATA_ROOT / "videos.json"
TRACK_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class MediaPaths:
    def subjects(self) -> Path:
        return MEDIA_SUBJECTS_ROOT

    def subject(self, subject_id: str) -> Path:
        return MEDIA_SUBJECTS_ROOT / subject_id

    def subject_image(self, subject_id: str, image_id: int) -> Path:
        return MEDIA_SUBJECTS_ROOT / subject_id / f"{image_id}.jpg"

    def frames(self, video_id: str) -> Path:
        return FRAMES_ROOT / video_id

    def frame(self, video_id: str, frame_idx: int) -> Path:
        return FRAMES_ROOT / video_id / f"{frame_idx}.jpg"

    def masked_crop(self, video_id: str, track_id: str, frame_idx: int) -> Path:
        return MASKED_CROPS_ROOT / video_id / track_id / f"{frame_idx}.png"


media_paths = MediaPaths()


class LocalFsMediaStore:
    def subject_image_hash(self, subject_id: str, image_id: int) -> str:
        return hashlib.sha256(
            self.subject_image_path(subject_id, image_id).read_bytes()
        ).hexdigest()

    def list_subjects(self) -> list[Subject]:
        root = media_paths.subjects()

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
        return media_paths.subject_image(subject_id, image_id)

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
    def subject_spot_region_reviews(self, subject_id: str, image_id: int) -> Path:
        return METADATA_SUBJECTS_ROOT / subject_id / SPOT_REGION_REVIEWS_DIR / f"{image_id}.json"

    def subject_spot_correspondences(self, subject_id: str, first_id: int, second_id: int) -> Path:
        return (
            METADATA_SUBJECTS_ROOT
            / subject_id
            / SPOT_CORRESPONDENCES_DIR
            / f"{first_id}-{second_id}.json"
        )

    def subject_mouth(self, subject_id: str, image_id: int) -> Path:
        return METADATA_SUBJECTS_ROOT / subject_id / MOUTHS_DIR / f"{image_id}.json"

    def subject_eye(self, subject_id: str, image_id: int) -> Path:
        return METADATA_SUBJECTS_ROOT / subject_id / EYES_DIR / f"{image_id}.json"

    def subject_spots(self, subject_id: str, image_id: int) -> Path:
        return METADATA_SUBJECTS_ROOT / subject_id / SPOTS_DIR / f"{image_id}.json"

    def bboxes(self) -> Path:
        return BBOXES_PATH

    def track(self, video_id: str) -> Path:
        return TRACKS_METADATA_ROOT / video_id / TRACK_METADATA_FILENAME

    def videos(self) -> Path:
        return VIDEOS_PATH


metadata_paths = MetadataPaths()


class LocalFsMetadataStore:
    def read_subject_spot_region_reviews(
        self, subject_id: str, image_id: int
    ) -> list[SpotRegionReview]:
        path = metadata_paths.subject_spot_region_reviews(subject_id, image_id)

        return read_json(path, default={"regions": []})["regions"]

    def write_subject_spot_region_reviews(
        self, subject_id: str, image_id: int, regions: list[SpotRegionReview]
    ) -> None:
        write_json_atomic(
            metadata_paths.subject_spot_region_reviews(subject_id, image_id),
            {
                "subject_id": subject_id,
                "image_id": image_id,
                "coordinate_system": "subject_image_pixels",
                "regions": regions,
            },
        )

    def read_bboxes(self) -> dict[str, BBoxMap]:
        path = metadata_paths.bboxes()

        return read_json(path, default={})

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
        path = metadata_paths.subject_spots(subject_id, image_id)

        data = read_json(path, default={"points": []})

        return data["points"]

    def write_subject_spots(self, subject_id: str, image_id: int, points: list[list[int]]) -> None:
        image_path = media_paths.subject_image(subject_id, image_id)
        write_json_atomic(
            metadata_paths.subject_spots(subject_id, image_id),
            {
                "subject_id": subject_id,
                "image_id": image_id,
                "image": str(image_path),
                "coordinate_system": "subject_image_pixels",
                "points": points,
            },
        )

    def read_subject_eye(self, subject_id: str, image_id: int) -> EyeAnnotation | None:
        path = metadata_paths.subject_eye(subject_id, image_id)

        payload = read_json(path, default={"eye": None})

        eye = payload["eye"]

        if eye is None:
            return None

        return EyeAnnotation(
            center=tuple(eye["center"]),
            radius_x=eye["radius_x"],
            radius_y=eye["radius_y"],
            angle_degrees=eye["angle_degrees"],
        )

    def write_subject_eye(self, subject_id: str, image_id: int, eye: EyeAnnotation) -> None:
        write_json_atomic(
            metadata_paths.subject_eye(subject_id, image_id),
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

    def read_subject_mouth(self, subject_id: str, image_id: int) -> tuple[float, float] | None:
        path = metadata_paths.subject_mouth(subject_id, image_id)

        data = read_json(path, default={"point": None})

        return tuple(data["point"]) if data["point"] is not None else None

    def write_subject_mouth(
        self, subject_id: str, image_id: int, point: tuple[float, float]
    ) -> None:
        write_json_atomic(
            metadata_paths.subject_mouth(subject_id, image_id),
            {
                "subject_id": subject_id,
                "image_id": image_id,
                "side": "left",
                "point": list(point),
            },
        )

    def read_subject_spot_correspondences(
        self, subject_id: str, first_id: int, second_id: int
    ) -> list[dict]:
        path = metadata_paths.subject_spot_correspondences(subject_id, first_id, second_id)

        payload = read_json(path, default={"pairs": []})

        return payload["pairs"]

    def write_subject_spot_correspondences(
        self,
        subject_id: str,
        first_id: int,
        second_id: int,
        pairs: list[dict],
    ) -> None:
        write_json_atomic(
            metadata_paths.subject_spot_correspondences(subject_id, first_id, second_id),
            {
                "subject_id": subject_id,
                "image_ids": [first_id, second_id],
                "coordinate_system": "subject_image_pixels",
                "pairs": pairs,
            },
        )
