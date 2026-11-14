import csv
from pathlib import Path

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
            global_inliers / matches
            if global_inliers is not None and matches
            else None
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

        writer.writerow(row) # pyright: ignore


def all_pairs(project: Project):
    from itertools import combinations, product

    subjects = list(project.subjects.get_all())

    for subject in subjects:
        images = [
            img for img in subject.images
            if img.binary_im is not None
        ]

        for a, b in combinations(images, 2):
            yield subject, subject, a, b

    for subject_a, subject_b in combinations(subjects, 2):
        images_a = [
            img for img in subject_a.images
            if img.binary_im is not None
        ]

        images_b = [
            img for img in subject_b.images
            if img.binary_im is not None
        ]

        for a, b in product(images_a, images_b):
            yield subject_a, subject_b, a, b
