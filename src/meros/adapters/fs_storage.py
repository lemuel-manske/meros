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
    FrameMedia,
    Individual,
    TrackSelectionKey,
    TrackSelection,
    MediaStore,
    MetadataStore,
    Track,
    TrackMetadata,
    TrackObservation,
    Video,
)
from meros.domain.storage import validate_bbox


@dataclass(frozen=True)
class MediaPaths:
    root: Path = Path("data")

    def video(self, video_id: str) -> Path:
        return (self.root / "media/videos") / f"{video_id}.mov"

    def frame(self, video_id: str, frame_idx: int) -> Path:
        return (self.root / "media/frames") / video_id / f"{frame_idx}.jpg"

    def masked_crop(self, video_id: str, track_id: str, frame_idx: int) -> Path:
        return (self.root / "media/masked_crops") / video_id / track_id / f"{frame_idx}.png"

    def aligned_overlay_crop(
        self, video_id: str, track_id: str, frame_idx: int, *, selection_id: str
    ) -> Path:
        return (
            (self.root / "media/alignment/selections") / selection_id / f"{frame_idx}_overlay.png"
        )

    def aligned_crop(
        self, video_id: str, track_id: str, frame_idx: int, *, selection_id: str
    ) -> Path:
        return (self.root / "media/alignment/selections") / selection_id / f"{frame_idx}.png"

    def composite(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> Path:
        return (self.root / "media/alignment/selections") / selection_id / "composite.png"

    def composite_enhanced(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> Path:
        return (self.root / "media/alignment/selections") / selection_id / "enhanced_composite.png"

    def composite_comparison(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> Path:
        return (self.root / "media/alignment/selections") / selection_id / "comparison.png"

    def visualization(self, video_id: str, frame_idx: int) -> Path:
        return (self.root / "media/visualizations") / video_id / f"{frame_idx}.jpg"


class LocalFsMediaStore(MediaStore):
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

    def has_frames(self, video_id: str) -> bool:
        return any(self.frames_path(video_id).glob("*.jpg"))

    def masked_crop_exists(self, video_id: str, track_id: str, frame_idx: int) -> bool:
        return self.paths.masked_crop(video_id, track_id, frame_idx).is_file()

    def aligned_crop_exists(
        self, video_id: str, track_id: str, frame_idx: int, *, selection_id: str
    ) -> bool:
        return self.paths.aligned_crop(
            video_id, track_id, frame_idx, selection_id=selection_id
        ).is_file()

    def composite_exists(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> bool:
        return self.paths.composite(
            video_id, track_id, reference_frame, selection_id=selection_id
        ).is_file()

    def composite_enhanced_exists(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> bool:
        return self.paths.composite_enhanced(
            video_id, track_id, reference_frame, selection_id=selection_id
        ).is_file()

    def composite_comparison_exists(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> bool:
        return self.paths.composite_comparison(
            video_id, track_id, reference_frame, selection_id=selection_id
        ).is_file()

    def visualization_exists(self, video_id: str, frame_idx: int) -> bool:
        return self.paths.visualization(video_id, frame_idx).is_file()

    def frames_path(self, video_id: str) -> Path:
        return (self.paths.root / "media/frames") / video_id

    def frame_path(self, video_id: str, frame_idx: int) -> Path:
        return self.paths.frame(video_id, frame_idx)

    def read_frame(self, video_id: str, frame_idx: int) -> np.ndarray:
        return self._read_image(self.paths.frame(video_id, frame_idx))

    def frame_ids(self, video_id: str) -> list[int]:
        return sorted(int(p.stem) for p in self.frames_path(video_id).glob("*.jpg"))

    def read_frames(self, video_id: str) -> list[FrameMedia]:
        parent = (self.paths.root / "media/frames") / video_id

        paths = sorted(parent.glob("*.jpg"), key=lambda path: int(path.stem))

        return [FrameMedia(frame_idx=int(path.stem), data=self._read_image(path)) for path in paths]

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

    def remove_alignment(
        self, video_id: str, track_id: str, frame_idx: int, *, selection_id: str
    ) -> None:
        self.paths.aligned_crop(video_id, track_id, frame_idx, selection_id=selection_id).unlink(
            missing_ok=True
        )

        self.paths.aligned_overlay_crop(
            video_id, track_id, frame_idx, selection_id=selection_id
        ).unlink(missing_ok=True)

    def read_aligned_crop(
        self, video_id: str, track_id: str, frame_idx: int, *, selection_id: str
    ) -> np.ndarray:
        return self._read_image(
            self.paths.aligned_crop(video_id, track_id, frame_idx, selection_id=selection_id),
            cv.IMREAD_UNCHANGED,
        )

    def write_aligned_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        aligned_crop: np.ndarray,
        *,
        selection_id: str,
    ) -> None:
        self._write_image(
            self.paths.aligned_crop(video_id, track_id, frame_idx, selection_id=selection_id),
            aligned_crop,
        )

    def write_aligned_overlay(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        aligned_overlay: np.ndarray,
        *,
        selection_id: str,
    ) -> None:
        self._write_image(
            self.paths.aligned_overlay_crop(
                video_id, track_id, frame_idx, selection_id=selection_id
            ),
            aligned_overlay,
        )

    def read_composite(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> np.ndarray:
        return self._read_image(
            self.paths.composite(video_id, track_id, reference_frame, selection_id=selection_id),
            cv.IMREAD_UNCHANGED,
        )

    def write_composite(
        self,
        video_id: str,
        track_id: str,
        reference_frame: int,
        composite: np.ndarray,
        *,
        selection_id: str,
    ) -> None:
        self._write_image(
            self.paths.composite(video_id, track_id, reference_frame, selection_id=selection_id),
            composite,
        )

    def read_composite_enhanced(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> np.ndarray:
        return self._read_image(
            self.paths.composite_enhanced(
                video_id, track_id, reference_frame, selection_id=selection_id
            ),
            cv.IMREAD_UNCHANGED,
        )

    def write_composite_enhanced(
        self,
        video_id: str,
        track_id: str,
        reference_frame: int,
        composite_enhanced: np.ndarray,
        *,
        selection_id: str,
    ) -> None:
        self._write_image(
            self.paths.composite_enhanced(
                video_id, track_id, reference_frame, selection_id=selection_id
            ),
            composite_enhanced,
        )

    def read_composite_comparison(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> np.ndarray:
        return self._read_image(
            self.paths.composite_comparison(
                video_id, track_id, reference_frame, selection_id=selection_id
            ),
            cv.IMREAD_UNCHANGED,
        )

    def write_composite_comparison(
        self,
        video_id: str,
        track_id: str,
        reference_frame: int,
        composite_comparison: np.ndarray,
        *,
        selection_id: str,
    ) -> None:
        self._write_image(
            self.paths.composite_comparison(
                video_id, track_id, reference_frame, selection_id=selection_id
            ),
            composite_comparison,
        )

    def read_visualization(self, video_id: str, frame_idx: int) -> np.ndarray:
        return self._read_image(self.paths.visualization(video_id, frame_idx))

    def write_visualization(self, video_id: str, frame_idx: int, visualization: np.ndarray) -> None:
        self._write_image(self.paths.visualization(video_id, frame_idx), visualization)


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


class LocalFsMetadataStore(MetadataStore):
    def __init__(self, root: Path = Path("data")) -> None:
        self.paths = MetadataPaths(root)

        self.individuals_path: Path | None = None

    def read_video(self, video_id: str) -> Video:
        videos = self.read_videos()

        for video in videos:
            if video.video_id == video_id:
                return video

        raise ValueError(f"Video not found: {video_id}")

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
        if self.individuals_path is None:
            raise ValueError("Choose a selection manifest before preparing representations")

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

        path.parent.mkdir(parents=True, exist_ok=True)

        write_json_atomic(path, asdict(metadata))

    def write_alignment(
        self, _id: TrackSelectionKey, reference_frame: int, rows: list[AlignmentMetadata]
    ) -> None:
        video_id, track_id, start_frame, end_frame, selected_reference = _id

        path = self.paths.alignment(video_id, track_id, start_frame, end_frame, selected_reference)

        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "reference_frame": reference_frame,
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
