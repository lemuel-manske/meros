from dataclasses import dataclass
from pathlib import Path

from meros.adapters import SubjectsAdapter, VideosAdapter
from meros.domain.storage import Subjects, Videos


@dataclass(frozen=True)
class Project:
    subjects: Subjects
    videos: Videos

    sam2_model: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt")
    yolo_model: Path = Path("external/yolo/yolo26n-cls.pt")
    fishial_model: Path = Path("external/yolo/fishial-detector.pt")

    """
    Finetuned Fishial model for detecting Goliath Groupers.
    """
    mero_detector: Path = Path("data/ml/detection/20261109-104626/weights/best.pt")

    @classmethod
    def open(
        cls,
    ) -> "Project":
        subjects = SubjectsAdapter()
        videos = VideosAdapter()

        return cls(subjects, videos)
