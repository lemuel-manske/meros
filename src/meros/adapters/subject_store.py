import csv
from datetime import date
from functools import partial
from pathlib import Path
import shutil

import cv2 as cv
import numpy as np

from meros.adapters.paths import media_paths, metadata_paths
from meros.domain.storage import Subject, SubjectImage, Subjects
from meros.utils import is_image_file, read_image, read_json, write_image, write_json

ANNOTATION_FIELDS = ["id", "capture_date", "source_video_id"]


def read_subject_crop(subject_id: str, image_id: int) -> np.ndarray:
    return read_image(media_paths.subject_crop(subject_id, image_id), cv.IMREAD_UNCHANGED)


def read_subject_binary(subject_id: str, image_id: int) -> np.ndarray | None:
    path = media_paths.subject_body(subject_id, image_id)

    return read_image(path, cv.IMREAD_UNCHANGED) if path.exists() else None


def read_subject_images(subject_id: str) -> list[SubjectImage]:
    directory = media_paths.subject(subject_id)

    if not directory.exists():
        return []

    with metadata_paths.subject_annotations(subject_id).open(newline="") as stream:
        annotations = {int(row["id"]): row for row in csv.DictReader(stream)}

    segmentation = read_json(metadata_paths.subject_segmentation(subject_id), default={})
    bboxes = segmentation.get("bboxes", {})

    images = []

    for path in directory.iterdir():
        if not is_image_file(path):
            continue

        image_id = int(path.stem)
        annotation = annotations[image_id]

        spots_path = metadata_paths.subject_spots(subject_id, image_id)
        spots = read_json(spots_path, default={"points": []})["points"]

        images.append(
            SubjectImage(
                _id=image_id,
                path=str(path),
                capture_date=date.fromisoformat(annotation["capture_date"]),
                source_video_id=annotation.get("source_video_id") or None,
                _load_im=partial(read_image, path),
                _load_crop_im=partial(read_subject_crop, subject_id, image_id),
                _load_binary_im=partial(read_subject_binary, subject_id, image_id),
                bbox=bboxes.get(str(image_id), []),
                spots=spots,
            )
        )

    return sorted(images, key=lambda image: image._id)


def write_image_if_changed(path: Path, image: np.ndarray) -> None:
    if path.exists() and np.array_equal(read_image(path, cv.IMREAD_UNCHANGED), image):
        return

    write_image(path, image)


def write_subject_images(subject: Subject) -> None:
    for image in subject.images:
        path = media_paths.subject_image(subject._id, image._id)

        if image.im_override is not None:
            write_image_if_changed(path, image.im_override)
        elif not path.exists():
            source = Path(image.path)

            if is_image_file(source):
                shutil.copyfile(source, path)
            else:
                write_image(path, image.im)

        if image.crop_override is not None:
            write_image_if_changed(
                media_paths.subject_crop(subject._id, image._id),
                image.crop_override,
            )

        if image.binary_override is not None:
            write_image_if_changed(
                media_paths.subject_body(subject._id, image._id),
                image.binary_override,
            )


def write_subject_metadata(subject: Subject) -> None:
    annotations_path = metadata_paths.subject_annotations(subject._id)

    annotations = {}

    if annotations_path.exists():
        with annotations_path.open(newline="") as stream:
            annotations = {int(row["id"]): row for row in csv.DictReader(stream)}

    updated_annotations = annotations.copy()

    segmentation_path = metadata_paths.subject_segmentation(subject._id)
    segmentation = read_json(segmentation_path, default={})
    bboxes = segmentation.get("bboxes", {}).copy()

    for image in subject.images:
        image_id = str(image._id)
        previous = annotations.get(image._id, {})

        updated_annotations[image._id] = {
            "id": image_id,
            "capture_date": image.capture_date.isoformat(),
            "source_video_id": image.source_video_id or previous.get("source_video_id", ""),
        }

        if image.bbox or image_id in bboxes:
            bboxes[image_id] = image.bbox

        spots_path = metadata_paths.subject_spots(subject._id, image._id)
        spots = {"points": image.spots}

        if (image.spots or spots_path.exists()) and spots != read_json(spots_path, default={}):
            write_json(spots_path, spots)

    if bboxes != segmentation.get("bboxes", {}):
        segmentation["bboxes"] = bboxes
        write_json(segmentation_path, segmentation)

    if updated_annotations != annotations:
        annotations_path.parent.mkdir(parents=True, exist_ok=True)

        with annotations_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(updated_annotations.values())


class SubjectsAdapter(Subjects):
    def get_all(self) -> list[Subject]:
        root = media_paths.subjects()

        if not root.exists():
            return []

        return [
            Subject(_id=directory.name, images=read_subject_images(directory.name))
            for directory in sorted(root.iterdir())
            if directory.is_dir()
        ]

    def get(self, subject_id: str) -> Subject:
        for subject in self.get_all():
            if subject._id == subject_id:
                return subject

        raise ValueError(f"Subject {subject_id} not found")

    def write(self, subject: Subject) -> None:
        media_paths.subject(subject._id).mkdir(parents=True, exist_ok=True)

        write_subject_images(subject)
        write_subject_metadata(subject)
