import cv2 as cv

import json
import os
import tempfile

import numpy as np

from dataclasses import asdict, dataclass
from pathlib import Path

from meros.domain import (
    AlignmentMetadata,
    AlignmentRunMetadata,
    BBox,
    Individual,
    TrackSelectionKey,
    TrackSelection,
    Track,
    TrackMetadata,
    TrackObservation,
    Video,
)
from meros.domain.storage import validate_bbox


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
    root: Path = Path("data")

    def frame(self, video_id: str, frame_idx: int) -> Path:
        return (self.root / "media/frames") / video_id / f"{frame_idx}.jpg"

    def masked_crop(self, video_id: str, track_id: str, frame_idx: int) -> Path:
        return (self.root / "media/masked_crops") / video_id / track_id / f"{frame_idx}.png"

    def composite(self, selection: TrackSelection) -> Path:
        return self.root / "media/alignment/selections" / selection.selection_id / "composite.png"

    def composite_enhanced(self, selection: TrackSelection) -> Path:
        return (
            self.root
            / "media/alignment/selections"
            / selection.selection_id
            / "enhanced_composite.png"
        )


class LocalFsMediaStore:
    def __init__(self, root: Path = Path("data")) -> None:
        self.paths = MediaPaths(root)

    @staticmethod
    def _read_image(path: Path, flags: int = cv.IMREAD_COLOR) -> np.ndarray:
        image = cv.imread(str(path), flags)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        return image

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if not cv.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")

    @staticmethod
    def valid_image(
        path: Path,
        channels: int,
        shape: tuple[int, int] | None = None,
        *,
        mask_area: int | None = None,
    ) -> bool:
        if not path.is_file():
            return False

        try:
            image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        except cv.error:
            return False

        if (
            image is None
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != channels
        ):
            return False

        if shape is not None and image.shape[:2] != shape:
            return False

        if channels == 4:
            alpha = image[:, :, 3]

            if not np.all((alpha == 0) | (alpha == 255)) or not np.any(alpha == 255):
                return False

            if mask_area is not None and np.count_nonzero(alpha) != mask_area:
                return False

        return True

    def frames_path(self, video_id: str) -> Path:
        return (self.paths.root / "media/frames") / video_id

    def frame_path(self, video_id: str, frame_idx: int) -> Path:
        return self.paths.frame(video_id, frame_idx)

    def read_frame(self, video_id: str, frame_idx: int) -> np.ndarray:
        return self._read_image(self.paths.frame(video_id, frame_idx))

    def frame_ids(self, video_id: str) -> list[int]:
        return sorted(int(p.stem) for p in self.frames_path(video_id).glob("*.jpg"))

    def write_frame(self, video_id: str, frame_idx: int, frame: np.ndarray) -> None:
        self._write_image(self.paths.frame(video_id, frame_idx), frame)

    def read_masked_crop(self, video_id: str, track_id: str, frame_idx: int) -> np.ndarray:
        return self._read_image(
            self.paths.masked_crop(video_id, track_id, frame_idx), cv.IMREAD_UNCHANGED
        )

    def write_masked_crop(
        self, video_id: str, track_id: str, frame_idx: int, masked_crop: np.ndarray
    ) -> None:
        self._write_image(self.paths.masked_crop(video_id, track_id, frame_idx), masked_crop)

    def read_composite(self, selection: TrackSelection) -> np.ndarray:
        return self._read_image(self.paths.composite(selection), cv.IMREAD_UNCHANGED)

    def write_composite(self, selection: TrackSelection, composite: np.ndarray) -> None:
        self._write_image(self.paths.composite(selection), composite)

    def read_composite_enhanced(self, selection: TrackSelection) -> np.ndarray:
        return self._read_image(self.paths.composite_enhanced(selection), cv.IMREAD_UNCHANGED)

    def write_composite_enhanced(self, selection: TrackSelection, composite: np.ndarray) -> None:
        self._write_image(self.paths.composite_enhanced(selection), composite)


@dataclass(frozen=True)
class MetadataPaths:
    root: Path = Path("data")

    def track(self, video_id: str) -> Path:
        return (self.root / "metadata/tracks") / video_id / "metadata.json"

    def alignment(
        self,
        video_id: str,
        track_id: str,
        start_frame: int,
        end_frame: int,
        reference_frame: int,
    ) -> Path:
        return (
            (self.root / "metadata/alignment")
            / video_id
            / f"{track_id}_{start_frame}_{end_frame}_ref_{reference_frame}.json"
        )

    def videos(self) -> Path:
        return (self.root / "metadata/videos") / "videos.json"

    def bboxes(self) -> Path:
        return (self.root / "metadata/videos") / "bboxes.json"


class LocalFsMetadataStore:
    def __init__(self, root: Path, manifest: Path) -> None:
        self.paths = MetadataPaths(root)

        self.individuals_path = manifest

    def read_bboxes(self) -> dict[str, BBox]:
        with self.paths.bboxes().open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict):
            raise ValueError("Seed boxes must be keyed by video and track")

        for tracks in metadata.values():
            if not isinstance(tracks, dict):
                raise ValueError("Seed boxes must be keyed by track")

            for bbox in tracks.values():
                validate_bbox(bbox)

        return metadata

    def read_videos(self) -> list[Video]:
        with self.paths.videos().open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return [Video(**video_metadata) for video_metadata in metadata]

    def read_individuals(self) -> list[Individual]:
        with self.individuals_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return [
            Individual(
                individual_id=item["individual_id"],
                tracks=[TrackSelection(**track) for track in item["tracks"]],
            )
            for item in metadata
        ]

    def read_track(self, video_id: str) -> TrackMetadata:
        path = self.paths.track(video_id)

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
        path = self.paths.track(video_id)

        write_json_atomic(path, asdict(metadata))

    def write_alignment(self, _id: TrackSelectionKey, rows: list[AlignmentMetadata]) -> None:
        video_id, track_id, start_frame, end_frame, selected_reference = _id

        path = self.paths.alignment(video_id, track_id, start_frame, end_frame, selected_reference)

        payload = {
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "reference_frame": selected_reference,
            "method": "SIFT mutual ratio matches, affine RANSAC",
            "frames": [asdict(row) for row in rows],
        }

        write_json_atomic(path, payload)

    def read_alignment(self, _id: TrackSelectionKey) -> AlignmentRunMetadata:
        video_id, track_id, start_frame, end_frame, selected_reference = _id

        path = self.paths.alignment(video_id, track_id, start_frame, end_frame, selected_reference)

        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata["reference_frame"] != selected_reference:
            raise ValueError("Alignment reference differs from selection")

        return AlignmentRunMetadata(
            video_id=metadata["video_id"],
            track_id=metadata["track_id"],
            start_frame=metadata["start_frame"],
            end_frame=metadata["end_frame"],
            reference_frame=metadata["reference_frame"],
            method=metadata["method"],
            frames=[AlignmentMetadata(**row) for row in metadata["frames"]],
        )
