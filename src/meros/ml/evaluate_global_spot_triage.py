"""
EXP-004K: screen global spot structure and defer uncertain identities to humans.

The current visual/geometric matcher remains unchanged. Provisional screening
bounds are not validated identity thresholds. No known identity is rejected
as a new fish automatically; uncertain comparisons are sent for review.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from meros.ml.evaluate_individual_reid import (
    MIN_SPOTS,
    Candidate,
    SpotPhoto,
    create_encoder,
    initial_similarity,
    load_photo,
    project_points,
    rank_individuals,
)
from meros.project import Project


MIN_OVERLAP_SMALL_HULL = 0.60
MIN_OVERLAP_LARGE_HULL = 0.35
MIN_SHARED_SPOTS = 8
BORDER_RATIO = 0.03
SPATIAL_GRID_SIZE = 3

MAX_DENSITY_RATIO = 1.60
MAX_SPATIAL_DIFFERENCE = 0.50
MIN_AUTO_SCORE = 0.22
MIN_AUTO_MARGIN = 0.08


@dataclass(frozen=True)
class GlobalProfile:
    overlap_small: float
    overlap_large: float
    source_spots: int
    target_spots: int
    density_ratio: float
    spatial_difference: float


def global_profile(source: SpotPhoto, target: SpotPhoto) -> GlobalProfile | None:
    """
    Compare the spatial distribution of annotated spots before affine refinement.

    Both sets are aligned only by eye/mouth landmarks. Convex hulls approximate
    annotated coverage, not confirmed visibility or complete annotation.
    """
    matrix = initial_similarity(source.eye, source.mouth, target.eye, target.mouth)
    projected = project_points(source.points, matrix).astype(np.float32)
    target_points = target.points.astype(np.float32)
    source_hull = cv.convexHull(projected)
    target_hull = cv.convexHull(target_points)
    source_area = cv.contourArea(source_hull)
    target_area = cv.contourArea(target_hull)

    if min(source_area, target_area) <= 0:
        return None

    intersection_area, intersection = cv.intersectConvexConvex(source_hull, target_hull)

    if intersection is None or intersection_area <= 0:
        return None

    overlap_small = float(intersection_area / min(source_area, target_area))
    overlap_large = float(intersection_area / max(source_area, target_area))

    if (
        overlap_small < MIN_OVERLAP_SMALL_HULL
        or overlap_large < MIN_OVERLAP_LARGE_HULL
    ):
        return None

    border = BORDER_RATIO * target.scale

    def shared_points(points: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                point
                for point in points
                if cv.pointPolygonTest(
                    intersection, (float(point[0]), float(point[1])), True
                )
                >= border
            ],
            dtype=np.float32,
        ).reshape(-1, 2)

    source_shared = shared_points(projected)
    target_shared = shared_points(target_points)

    if min(len(source_shared), len(target_shared)) < MIN_SHARED_SPOTS:
        return None

    density_ratio = max(len(source_shared), len(target_shared)) / min(
        len(source_shared), len(target_shared)
    )

    # Use identical anatomical-aligned cells within the shared footprint.
    vertices = intersection.reshape(-1, 2)
    minimum = vertices.min(axis=0)
    span = vertices.max(axis=0) - minimum

    if np.any(span <= 0):
        return None

    def distribution(points: np.ndarray) -> np.ndarray:
        normalized = np.clip((points - minimum) / span, 0.0, 1.0 - 1e-9)
        histogram, _, _ = np.histogram2d(
            normalized[:, 0],
            normalized[:, 1],
            bins=SPATIAL_GRID_SIZE,
            range=((0.0, 1.0), (0.0, 1.0)),
        )

        return histogram / len(points)

    spatial_difference = float(
        0.5 * np.abs(distribution(source_shared) - distribution(target_shared)).sum()
    )

    return GlobalProfile(
        overlap_small=overlap_small,
        overlap_large=overlap_large,
        source_spots=len(source_shared),
        target_spots=len(target_shared),
        density_ratio=float(density_ratio),
        spatial_difference=spatial_difference,
    )


def triage(
    query: SpotPhoto,
    ranking: list[Candidate],
    gallery: dict[tuple[str, int], SpotPhoto],
) -> tuple[str, str, GlobalProfile | None]:
    """
    Accept only strong, separated, structurally compatible candidates.

    Every uncertain case is referred to a human; no unknown-identity verdict
    is made automatically from these uncalibrated experimental thresholds.
    """
    if not ranking:
        return "REVIEW", "no-candidates", None

    best = ranking[0]
    reference = gallery[(best.subject_id, best.image_id)]
    profile = global_profile(query, reference)
    reasons = []

    if best.score < MIN_AUTO_SCORE:
        reasons.append("weak-score")

    if len(ranking) > 1 and best.score - ranking[1].score < MIN_AUTO_MARGIN:
        reasons.append("close-candidates")

    if profile is None:
        reasons.append("insufficient-shared-annotation")
    else:
        if profile.density_ratio > MAX_DENSITY_RATIO:
            reasons.append("density-mismatch")

        if profile.spatial_difference > MAX_SPATIAL_DIFFERENCE:
            reasons.append("distribution-mismatch")

    if reasons:
        return "REVIEW", ",".join(reasons), profile

    return "AUTO", "provisionally-consistent", profile


def run(project: Project) -> None:
    """
    Evaluate selective identification with known and fully held-out identities.
    """
    encoder = create_encoder()
    gallery: dict[tuple[str, int], SpotPhoto] = {}

    for subject in project.media.list_subjects():
        for image in subject.images:
            label = (subject.subject_id, image.image_id)

            if len(project.metadata.read_subject_spots(*label)) < MIN_SPOTS:
                continue

            photo = load_photo(project, label, encoder)

            if photo is not None:
                gallery[label] = photo

    counts = Counter(subject for subject, _ in gallery)
    known_queries = [label for label in gallery if counts[label[0]] >= 2]
    score_cache: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}

    print(f"Gallery: {len(gallery)} images, {len(counts)} identities")
    print(
        f"EXP-004K exploratory triage: min_score={MIN_AUTO_SCORE:.2f}, "
        f"min_margin={MIN_AUTO_MARGIN:.2f}; "
        f"max_density_ratio={MAX_DENSITY_RATIO:.2f}, "
        f"max_spatial_difference={MAX_SPATIAL_DIFFERENCE:.2f}"
    )
    print(
        "Global evidence uses only eye-mouth alignment, prior to fitted affine "
        "matching. REVIEW is abstention, not a new-individual decision."
    )

    for mode, labels in (("known", known_queries), ("unknown", list(gallery))):
        n_auto = 0
        n_correct_auto = 0
        n_false_auto = 0
        n_correct_baseline = 0

        print(f"\\n{mode.upper()}: {len(labels)} queries")
        print(
            "query | best | score | margin | spot_counts | density_ratio "
            "| spatial_difference | decision | reason"
        )

        for label in labels:
            query = gallery[label]
            ranking = rank_individuals(
                label,
                query,
                gallery,
                score_cache,
                excluded_subject=label[0] if mode == "unknown" else None,
            )

            if not ranking:
                print(f"{label[0]}/{label[1]} | no candidates | REVIEW")
                continue

            best = ranking[0]
            margin = best.score - ranking[1].score if len(ranking) > 1 else 0.0
            decision, reason, profile = triage(query, ranking, gallery)
            correct = mode == "known" and best.subject_id == label[0]
            n_correct_baseline += int(correct)

            if decision == "AUTO":
                n_auto += 1
                n_correct_auto += int(correct)
                n_false_auto += int(not correct)

            counts_text = (
                f"{profile.source_spots}+{profile.target_spots}"
                if profile is not None else "n/a"
            )
            ratio_text = f"{profile.density_ratio:.2f}" if profile else "n/a"
            spatial_text = f"{profile.spatial_difference:.2f}" if profile else "n/a"

            print(
                f"{label[0]}/{label[1]} | {best.subject_id}/{best.image_id} | "
                f"{best.score:.3f} | {margin:.3f} | {counts_text} | "
                f"{ratio_text} | {spatial_text} | {decision} | {reason}"
            )

        if mode == "known":
            print(
                f"Known: baseline top-1={n_correct_baseline}/{len(labels)}, "
                f"AUTO={n_auto}/{len(labels)}, correct AUTO={n_correct_auto}, "
                f"wrong AUTO={n_false_auto}, REVIEW={len(labels) - n_auto}"
            )
        else:
            print(
                f"Unknown: falsely AUTO={n_false_auto}/{len(labels)}, "
                f"REVIEW={len(labels) - n_auto}/{len(labels)}"
            )

    print(
        "Thresholds are illustrative and chosen without held-out calibration. "
        "Appearance and spot annotation coverage can change over time. "
        "Do not interpret AUTO as a validated identity confidence."
    )


if __name__ == "__main__":
    run(Project.open())
