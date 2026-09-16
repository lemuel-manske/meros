import csv
import cv2 as cv
import json
import numpy as np

from dataclasses import asdict, dataclass
from pathlib import Path

from src.meros.domain import (
    AlignmentMetadata,
    AlignmentRunMetadata,
    FrameMedia,
    Individual,
    IndividualTrack,
    MediaStore,
    MetadataStore,
    Track,
    TrackAnnotation,
    TrackMetadata,
    TrackObservation,
    Video,
)


_DATA_FOLDER = Path("data")

_MEDIA_FOLDER = _DATA_FOLDER / "media"

_ALIGNMENT_MEDIA_FOLDER = _MEDIA_FOLDER / "alignment"
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
    def video(
        self,
        video_id: str,
    ) -> Path:
        return _VIDEOS_MEDIA_FOLDER / f"{video_id}.mov"

    def frame(
        self,
        video_id: str,
        frame_idx: int,
    ) -> Path:
        return _FRAMES_MEDIA_FOLDER / video_id / f"{frame_idx}.jpg"

    def crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> Path:
        return _CROPS_MEDIA_FOLDER / video_id / track_id / f"{frame_idx}.jpg"

    def masked_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> Path:
        return (
            _MASKED_CROPS_MEDIA_FOLDER
            / video_id
            / track_id
            / f"{frame_idx}.png"
        )

    def aligned_overlay_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> Path:
        return (
            _ALIGNMENT_MEDIA_FOLDER
            / video_id
            / track_id
            / f"{frame_idx}_overlay.png"
        )

    def aligned_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> Path:
        return (
            _ALIGNMENT_MEDIA_FOLDER
            / video_id
            / track_id
            / f"{frame_idx}.png"
        )

    def composite_enhanced(
        self,
        video_id: str,
        track_id: str,
    ) -> Path:
        return (
            _ALIGNMENT_MEDIA_FOLDER
            / video_id
            / track_id
            / "median_enhanced.png"
        )

    def composite(
        self,
        video_id: str,
        track_id: str,
    ) -> Path:
        return (
            _ALIGNMENT_MEDIA_FOLDER
            / video_id
            / track_id
            / "median.png"
        )

    def composite_comparison(
        self,
        video_id: str,
        track_id: str,
    ) -> Path:
        return (
            _ALIGNMENT_MEDIA_FOLDER
            / video_id
            / track_id
            / "median_comparison.png"
        )

    def composite_matches(
        self,
        video_id: str,
        track_id: str,
        other_video_id: str,
        other_track_id: str,
        enhanced: bool = False,
    ) -> Path:
        variant = "enhanced" if enhanced else "original"

        pair = f"{video_id}_{track_id}--{other_video_id}_{other_track_id}"

        return _MATCHES_MEDIA_FOLDER / pair / f"{variant}.png"

    def visualization(
        self,
        video_id: str,
        frame_idx: int,
    ) -> Path:
        return (
            _VISUALIZATIONS_MEDIA_FOLDER
            / video_id
            / f"{frame_idx}.jpg"
        )


class LocalFsMediaStore(MediaStore):
    def __init__(self) -> None:
        self.paths = MediaPaths()

    @staticmethod
    def _read_image(
        path: Path,
        flags: int = cv.IMREAD_COLOR,
    ) -> np.ndarray:
        image = cv.imread(str(path), flags)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        return image

    @staticmethod
    def _write_image(
        path: Path,
        image: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if not cv.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")

    def has_frames(
        self,
        video_id: str,
    ) -> bool:
        return any(
            self.frames_path(video_id).glob("*.jpg")
        )

    def masked_crop_exists(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> bool:
        return self.paths.masked_crop(
            video_id,
            track_id,
            frame_idx,
        ).is_file()

    def aligned_crop_exists(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> bool:
        return self.paths.aligned_crop(
            video_id,
            track_id,
            frame_idx,
        ).is_file()

    def composite_exists(
        self,
        video_id: str,
        track_id: str,
    ) -> bool:
        return self.paths.composite(
            video_id,
            track_id,
        ).is_file()

    def composite_enhanced_exists(
        self,
        video_id: str,
        track_id: str,
    ) -> bool:
        return self.paths.composite_enhanced(
            video_id,
            track_id,
        ).is_file()

    def composite_comparison_exists(
        self,
        video_id: str,
        track_id: str,
    ) -> bool:
        return self.paths.composite_comparison(
            video_id,
            track_id,
        ).is_file()

    def composite_matches_exists(
        self,
        video_id: str,
        track_id: str,
        other_video_id: str,
        other_track_id: str,
        enhanced: bool = False,
    ) -> bool:
        return self.paths.composite_matches(
            video_id,
            track_id,
            other_video_id,
            other_track_id,
            enhanced,
        ).is_file()

    def frames_path(
        self,
        video_id: str,
    ) -> Path:
        return _FRAMES_MEDIA_FOLDER / video_id

    def read_frame(
        self,
        video_id: str,
        frame_idx: int,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.frame(video_id, frame_idx)
        )

    def read_frames(
        self,
        video_id: str,
    ) -> list[FrameMedia]:
        parent = _FRAMES_MEDIA_FOLDER / video_id

        paths = sorted(
            parent.glob("*.jpg"),
            key=lambda path: int(path.stem),
        )

        return [
            FrameMedia(
                frame_idx=int(path.stem),
                data=self._read_image(path),
            )
            for path in paths
        ]

    def write_frame(
        self,
        video_id: str,
        frame_idx: int,
        frame: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.frame(video_id, frame_idx),
            frame,
        )

    def read_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.crop(video_id, track_id, frame_idx)
        )

    def write_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        crop: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.crop(video_id, track_id, frame_idx),
            crop,
        )

    def read_masked_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.masked_crop(video_id, track_id, frame_idx),
            cv.IMREAD_UNCHANGED,
        )

    def write_masked_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        masked_crop: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.masked_crop(video_id, track_id, frame_idx),
            masked_crop,
        )

    def remove_alignment(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> None:
        self.paths.aligned_crop(
            video_id,
            track_id,
            frame_idx,
        ).unlink(missing_ok=True)

        self.paths.aligned_overlay_crop(
            video_id,
            track_id,
            frame_idx,
        ).unlink(missing_ok=True)

    def read_aligned_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.aligned_crop(video_id, track_id, frame_idx),
            cv.IMREAD_UNCHANGED,
        )

    def write_aligned_crop(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        aligned_crop: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.aligned_crop(video_id, track_id, frame_idx),
            aligned_crop,
        )

    def write_aligned_overlay(
        self,
        video_id: str,
        track_id: str,
        frame_idx: int,
        aligned_overlay: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.aligned_overlay_crop(
                video_id,
                track_id,
                frame_idx,
            ),
            aligned_overlay,
        )

    def read_composite(
        self,
        video_id: str,
        track_id: str,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.composite(video_id, track_id),
            cv.IMREAD_UNCHANGED,
        )

    def write_composite(
        self,
        video_id: str,
        track_id: str,
        composite: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.composite(video_id, track_id),
            composite,
        )

    def read_composite_enhanced(
        self,
        video_id: str,
        track_id: str,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.composite_enhanced(video_id, track_id),
            cv.IMREAD_UNCHANGED,
        )

    def write_composite_enhanced(
        self,
        video_id: str,
        track_id: str,
        composite_enhanced: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.composite_enhanced(video_id, track_id),
            composite_enhanced,
        )

    def read_composite_comparison(
        self,
        video_id: str,
        track_id: str,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.composite_comparison(video_id, track_id),
            cv.IMREAD_UNCHANGED,
        )

    def write_composite_comparison(
        self,
        video_id: str,
        track_id: str,
        composite_comparison: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.composite_comparison(video_id, track_id),
            composite_comparison,
        )

    def read_composite_matches(
        self,
        video_id: str,
        track_id: str,
        other_video_id: str,
        other_track_id: str,
        enhanced: bool = False,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.composite_matches(
                video_id,
                track_id,
                other_video_id,
                other_track_id,
                enhanced,
            )
        )

    def write_composite_matches(
        self,
        video_id: str,
        track_id: str,
        other_video_id: str,
        other_track_id: str,
        composite_matches: np.ndarray,
        enhanced: bool = False,
    ) -> None:
        self._write_image(
            self.paths.composite_matches(
                video_id,
                track_id,
                other_video_id,
                other_track_id,
                enhanced,
            ),
            composite_matches,
        )

    def read_visualization(
        self,
        video_id: str,
        frame_idx: int,
    ) -> np.ndarray:
        return self._read_image(
            self.paths.visualization(video_id, frame_idx)
        )

    def write_visualization(
        self,
        video_id: str,
        frame_idx: int,
        visualization: np.ndarray,
    ) -> None:
        self._write_image(
            self.paths.visualization(video_id, frame_idx),
            visualization,
        )


@dataclass(frozen=True)
class MetadataPaths:
    def track(
        self,
        video_id: str,
    ) -> Path:
        return (
            _TRACKS_METADATA_FOLDER
            / video_id
            / "metadata.json"
        )

    def annotations(
        self,
        video_id: str,
        track_id: str,
    ) -> Path:
        return (
            _ANNOTATIONS_METADATA_FOLDER
            / video_id
            / track_id
            / "metadata.csv"
        )

    def alignment(
        self,
        video_id: str,
        track_id: str,
    ) -> Path:
        return (
            _ALIGNMENT_METADATA_FOLDER
            / video_id
            / f"{track_id}.json"
        )

    def videos(self) -> Path:
        return _VIDEOS_METADATA_FOLDER / "videos.json"

    def individuals(self) -> Path:
        return _INDIVIDUALS_METADATA_FOLDER / "individuals.json"


class LocalFsMetadataStore(MetadataStore):
    def __init__(self) -> None:
        self.paths = MetadataPaths()

    def read_videos(
        self,
    ) -> list[Video]:
        with self.paths.videos().open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return [
            Video(**video_metadata)
            for video_metadata in metadata
        ]

    def read_individuals(
        self,
    ) -> list[Individual]:
        with self.paths.individuals().open(
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        return [
            Individual(
                individual_id=item["individual_id"],
                tracks=[
                    IndividualTrack(**track)
                    for track in item["tracks"]
                ],
            )
            for item in metadata
        ]

    def read_track(
        self,
        video_id: str,
    ) -> TrackMetadata:
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
        )

    def write_track(
        self,
        video_id: str,
        metadata: TrackMetadata,
    ) -> None:
        path = self.paths.track(video_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                asdict(metadata),
                f,
                indent=2,
            )
            f.write("\n")

    def read_annotations(
        self,
        video_id: str,
        track_id: str,
    ) -> list[TrackAnnotation]:
        path = self.paths.annotations(video_id, track_id)

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)

            return [
                TrackAnnotation(
                    track_id=row["track_id"],
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    viewpoint=row["viewpoint"],
                    quality=int(row["quality"]),
                )
                for row in reader
            ]

    def write_alignment(
        self,
        video_id: str,
        track_id: str,
        start_frame: int,
        end_frame: int,
        reference_frame: int,
        rows: list[AlignmentMetadata],
    ) -> None:
        path = self.paths.alignment(video_id, track_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "video_id": video_id,
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "reference_frame": reference_frame,
            "method": "SIFT mutual ratio matches, affine RANSAC",
            "frames": [
                asdict(row)
                for row in rows
            ],
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def read_alignment(
        self,
        video_id: str,
        track_id: str,
    ) -> AlignmentRunMetadata:
        path = self.paths.alignment(video_id, track_id)

        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return AlignmentRunMetadata(
            video_id=metadata["video_id"],
            track_id=metadata["track_id"],
            start_frame=metadata["start_frame"],
            end_frame=metadata["end_frame"],
            reference_frame=metadata["reference_frame"],
            method=metadata["method"],
            frames=[
                AlignmentMetadata(**row)
                for row in metadata["frames"]
            ],
        )
