"""
Prepare a leave-one-image-out gallery for geometric individual re-identification.

Only manually marked spot centers are used. Verified spot correspondences are
reserved for evaluation and are never supplied to an identifier.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from meros.project import Project


MIN_SPOTS = 4


@dataclass(frozen=True)
class SpotImage:
    subject_id: str
    image_id: int
    spots: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.subject_id}/{self.image_id}"


def load_gallery(project: Project) -> list[SpotImage]:
    gallery = []

    for subject in project.media.list_subjects():
        for image in subject.images:
            points = project.metadata.read_subject_spots(subject.subject_id, image.image_id)

            if len(points) < MIN_SPOTS:
                continue

            gallery.append(
                SpotImage(
                    subject_id=subject.subject_id,
                    image_id=image.image_id,
                    spots=np.asarray(points, dtype=np.float64).reshape(-1, 2),
                )
            )

    return gallery


def evaluation_queries(gallery: list[SpotImage]) -> list[SpotImage]:
    counts = Counter(image.subject_id for image in gallery)

    return [image for image in gallery if counts[image.subject_id] >= 2]


def gallery_for_query(query: SpotImage, gallery: list[SpotImage]) -> list[SpotImage]:
    return [
        image
        for image in gallery
        if (image.subject_id, image.image_id) != (query.subject_id, query.image_id)
    ]


def run(project: Project) -> None:
    gallery = load_gallery(project)
    queries = evaluation_queries(gallery)
    identities = sorted({image.subject_id for image in gallery})

    print(f"Gallery: {len(gallery)} images from {len(identities)} individuals")
    print(f"Eligible queries: {len(queries)}")
    print()

    for subject_id in identities:
        images = [image for image in gallery if image.subject_id == subject_id]
        labels = ", ".join(image.label for image in images)
        print(f"  {subject_id}: {len(images)} image(s) [{labels}]")

    print("\nLeave-one-image-out evaluation:")

    for query in queries:
        candidates = gallery_for_query(query, gallery)
        positive = sum(image.subject_id == query.subject_id for image in candidates)
        negative = len(candidates) - positive

        print(
            f"  {query.label}: {len(candidates)} candidates "
            f"({positive} positive, {negative} negative)"
        )

    if not queries:
        print("No eligible queries: at least two annotated images per individual are needed.")


if __name__ == "__main__":
    run(Project.open())
