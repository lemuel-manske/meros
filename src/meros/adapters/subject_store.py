import csv
from datetime import date
from functools import partial

import cv2 as cv
import numpy as np

from meros.adapters.paths import media_paths, metadata_paths
from meros.domain.storage import (
    BBox,
    Subject,
    SubjectImage,
    Subjects,
    SubjectSpot,
)
from meros.utils import is_image_file, read_image, read_json


def read_subject_images(subject_id: str) -> list[SubjectImage]:
    directory = media_paths.subject(subject_id)

    if not directory.exists():
        return []

    def read_subject_capture_date(subject_id: str, frame_idx: int) -> date:
        path = metadata_paths.subject_annotations(subject_id)

        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["id"]) == frame_idx:
                    return date.fromisoformat(row["capture_date"])

        assert False, f"Frame {frame_idx} not found in {path}"

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

    def read_subject_binary(subject_id: str, image_id: int) -> np.ndarray | None:
        p = media_paths.subject_body(subject_id, image_id)

        if not p.exists():
            return None

        return read_image(p, cv.IMREAD_UNCHANGED)

    records = [
        SubjectImage(
            _id=int(path.stem),
            capture_date=read_subject_capture_date(subject_id, int(path.stem)),
            path=str(path),
            _load_im=partial(read_image, path),
            _load_crop_im=partial(read_subject_crop, subject_id, int(path.stem)),
            _load_binary_im=partial(read_subject_binary, subject_id, int(path.stem)),
            bbox=read_subject_bbox(subject_id, int(path.stem)),
            spots=read_subject_spots(subject_id, int(path.stem)),
        )
        for path in directory.iterdir()
        if is_image_file(path)
    ]

    return sorted(records, key=lambda item: item._id)


class SubjectsAdapter(Subjects):
    def get_all(self) -> list[Subject]:
        root = media_paths.subjects()

        if not root.exists():
            return []

        return [
            Subject(
                _id=directory.name,
                images=read_subject_images(directory.name),
            )
            for directory in sorted(root.iterdir())
            if directory.is_dir()
        ]

    def get(self, subject_id: str) -> Subject:
        for subject in self.get_all():
            if subject._id == subject_id:
                return subject

        raise ValueError(f"Subject {subject_id} not found")

    def write(self, subject: Subject) -> None: ...
