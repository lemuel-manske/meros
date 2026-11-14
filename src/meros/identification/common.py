import csv
from pathlib import Path
from typing import Generator

from meros.domain.storage import Subject, SubjectImage
from meros.project import Project

FIELDS = (
    "method",
    "subject_a",
    "image_a",
    "subject_b",
    "image_b",
    "same_individual",
    "status",
    "matches",
    "global_inliers",
    "global_inlier_ratio",
    "local_inliers",
    "both_inliers",
    "median_error",
    "coverage_a",
    "coverage_b",
    "pattern_iou",
    "inlier_coverage",
)


def save_matches(
    path: Path,
    *,
    method: str,
    subject_a: str,
    image_a: str | int,
    subject_b: str,
    image_b: str | int,
    status: str,
    matches: int | None = None,
    global_inliers: int | None = None,
    local_inliers: int | None = None,
    both_inliers: int | None = None,
    median_error: float | None = None,
    coverage_a: float | None = None,
    coverage_b: float | None = None,
    pattern_iou: float | None = None,
    inlier_coverage: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "method": method,
        "subject_a": subject_a,
        "image_a": image_a,
        "subject_b": subject_b,
        "image_b": image_b,
        "same_individual": subject_a == subject_b,
        "status": status,
        "matches": matches,
        "global_inliers": global_inliers,
        "global_inlier_ratio": (
            global_inliers / matches if global_inliers is not None and matches else None
        ),
        "local_inliers": local_inliers,
        "both_inliers": both_inliers,
        "median_error": median_error,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "pattern_iou": pattern_iou,
        "inlier_coverage": inlier_coverage,
    }

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(row)  # pyright: ignore


def pairs(
    project: Project,
    a_filter: str | None = None,
    b_filter: str | None = None,
    bin_filter: bool = True,
) -> Generator[tuple[Subject, Subject, SubjectImage, SubjectImage], None, None]:
    from itertools import combinations

    images = [
        (subject, image)
        for subject in project.subjects.get_all()
        for image in subject.images
        if (not bin_filter or image.binary_im is None)
    ]

    for (subject_a, a), (subject_b, b) in combinations(images, 2):
        forward = (a_filter is None or subject_a._id == a_filter) and (
            b_filter is None or subject_b._id == b_filter
        )

        reverse = (a_filter is None or subject_b._id == a_filter) and (
            b_filter is None or subject_a._id == b_filter
        )

        if forward:
            yield subject_a, subject_b, a, b
        elif reverse:
            yield subject_b, subject_a, b, a
