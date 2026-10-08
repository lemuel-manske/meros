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

_DATA_FOLDER = Path("data")
_MEDIA_FOLDER = _DATA_FOLDER / "media"
_ALIGNMENT_MEDIA_FOLDER = _MEDIA_FOLDER / "alignment"
_SELECTION_MEDIA_FOLDER = _ALIGNMENT_MEDIA_FOLDER / "selections"
_CROPS_MEDIA_FOLDER = _MEDIA_FOLDER / "crops"
_FRAMES_MEDIA_FOLDER = _MEDIA_FOLDER / "frames"
_MASKED_CROPS_MEDIA_FOLDER = _MEDIA_FOLDER / "masked_crops"
_MATCHES_MEDIA_FOLDER = _MEDIA_FOLDER / "matches"
_VIDEOS_MEDIA_FOLDER = _MEDIA_FOLDER / "videos"
_VISUALIZATIONS_MEDIA_FOLDER = _MEDIA_FOLDER / "visualizations"
_METADATA_FOLDER = _DATA_FOLDER / "metadata"
_ALIGNMENT_METADATA_FOLDER = _METADATA_FOLDER / "alignment"
_ANNOTATIONS_METADATA_FOLDER = _METADATA_FOLDER / "annotations"
_INDIVIDUALS_METADATA_FOLDER = _METADATA_FOLDER / "individuals"
_TRACKS_METADATA_FOLDER = _METADATA_FOLDER / "tracks"
_VIDEOS_METADATA_FOLDER = _METADATA_FOLDER / "videos"


@dataclass(frozen=True)
class MediaPaths:
    def video(self, video_id: str) -> Path:
        return _VIDEOS_MEDIA_FOLDER / f"{video_id}.mov"

    def frame(self, video_id: str, frame_idx: int) -> Path:
        return _FRAMES_MEDIA_FOLDER / video_id / f"{frame_idx}.jpg"

    def masked_crop(self, video_id: str, track_id: str, frame_idx: int) -> Path:
        return _MASKED_CROPS_MEDIA_FOLDER / video_id / track_id / f"{frame_idx}.png"

    def aligned_overlay_crop(
        self, video_id: str, track_id: str, frame_idx: int, *, selection_id: str
    ) -> Path:
        return _SELECTION_MEDIA_FOLDER / selection_id / f"{frame_idx}_overlay.png"

    def aligned_crop(
        self, video_id: str, track_id: str, frame_idx: int, *, selection_id: str
    ) -> Path:
        return _SELECTION_MEDIA_FOLDER / selection_id / f"{frame_idx}.png"

    def composite(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> Path:
        return _SELECTION_MEDIA_FOLDER / selection_id / "composite.png"

    def composite_enhanced(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> Path:
        return _SELECTION_MEDIA_FOLDER / selection_id / "enhanced_composite.png"

    def composite_comparison(
        self, video_id: str, track_id: str, reference_frame: int, *, selection_id: str
    ) -> Path:
        return _SELECTION_MEDIA_FOLDER / selection_id / "comparison.png"

    def visualization(self, video_id: str, frame_idx: int) -> Path:
        return _VISUALIZATIONS_MEDIA_FOLDER / video_id / f"{frame_idx}.jpg"


class LocalFsMediaStore(MediaStore):
    def __init__(self) -> None:
        self.paths = MediaPaths()

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
        return _FRAMES_MEDIA_FOLDER / video_id

    def frame_path(self, video_id: str, frame_idx: int) -> Path:
        return self.paths.frame(video_id, frame_idx)

    def read_frame(self, video_id: str, frame_idx: int) -> np.ndarray:
        return self._read_image(self.paths.frame(video_id, frame_idx))

    def frame_ids(self, video_id: str) -> list[int]:
        return sorted(int(p.stem) for p in self.frames_path(video_id).glob("*.jpg"))

    def read_frames(self, video_id: str) -> list[FrameMedia]:
        parent = _FRAMES_MEDIA_FOLDER / video_id

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
    def track(self, video_id: str) -> Path:
        return _TRACKS_METADATA_FOLDER / video_id / "metadata.json"

    def alignment(
        self,
        video_id: str,
        track_id: str,
        start_frame: int,
        end_frame: int,
        reference_frame: int,
    ) -> Path:
        return (
            _ALIGNMENT_METADATA_FOLDER
            / video_id
            / f"{track_id}_{start_frame}_{end_frame}_ref_{reference_frame}.json"
        )

    def videos(self) -> Path:
        return _VIDEOS_METADATA_FOLDER / "videos.json"

    def bboxes(self) -> Path:
        return _VIDEOS_METADATA_FOLDER / "bboxes.json"

    def individuals(self) -> Path:
        return _INDIVIDUALS_METADATA_FOLDER / "individuals.json"


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
    def __init__(self) -> None:
        self.paths = MetadataPaths()

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

        return metadata

    def read_videos(self) -> list[Video]:
        with self.paths.videos().open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return [Video(**video_metadata) for video_metadata in metadata]

    def read_individuals(self) -> list[Individual]:
        with (self.individuals_path or self.paths.individuals()).open("r", encoding="utf-8") as f:
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
