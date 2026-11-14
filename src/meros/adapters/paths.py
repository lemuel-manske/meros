from pathlib import Path

"""
Always extract constants for paths and directories.
"""

METADATA_ROOT = Path("data/metadata")

METADATA_SUBJECTS_ROOT = METADATA_ROOT / "subjects"
VIDEOS_METADATA_ROOT = METADATA_ROOT / "videos"

VIDEO_DETECTIONS_ROOT = VIDEOS_METADATA_ROOT / "detections"
VIDEO_PREDICTIONS_ROOT = VIDEOS_METADATA_ROOT / "predictions"


MEDIA_ROOT = Path("data/media")
MEDIA_SUBJECTS_ROOT = MEDIA_ROOT / "subjects"
VIDEOS_ROOT = MEDIA_ROOT / "videos"

SELECTED_VIDEO_FRAMES_ROOT = VIDEOS_ROOT / "selected"


class MediaPaths:
    def videos(self) -> Path:
        return VIDEOS_ROOT

    def selected_video_frames(self, video_id: str) -> Path:
        return SELECTED_VIDEO_FRAMES_ROOT / video_id

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
