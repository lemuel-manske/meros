"""
Inspect whether high-scoring matches require anatomically implausible affine changes.

This is a diagnostic only. It does not change identification scores or rankings.
The provisional bounds below are not physical laws or calibrated thresholds.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist

from meros.ml.evaluate_individual_reid import (
    GEOMETRIC_RADIUS_RATIO,
    MIN_SPOTS,
    SpotPhoto,
    create_encoder,
    geometric_visual_match,
    initial_similarity,
    load_photo,
    project_points,
    rank_individuals,
)
from meros.project import Project


MAX_ANISOTROPY = 1.8
MIN_RELATIVE_SCALE = 0.7
MAX_RELATIVE_SCALE = 1.4
MAX_ANCHOR_DRIFT = 0.12
MIN_ANNOTATED_OVERLAP = 0.20
ANNOTATED_BORDER_RATIO = 0.03
MIN_SHARED_SPOTS = 5
UNSUPPORTED_RADIUS_RATIO = GEOMETRIC_RADIUS_RATIO


def missing_spot_evidence(
    source: SpotPhoto,
    target: SpotPhoto,
    matrix: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> dict | None:
    """
    Estimate unaccounted spots inside intersecting annotation footprints.

    Convex hulls of manually annotated spots approximate comparable regions.
    They do not establish that the skin is visible or exhaustively annotated.
    """
    projected = project_points(source.points, matrix)
    source_hull = cv.convexHull(projected.astype(np.float32))
    target_hull = cv.convexHull(target.points.astype(np.float32))
    source_area = cv.contourArea(source_hull)
    target_area = cv.contourArea(target_hull)

    if min(source_area, target_area) <= 0:
        return None

    area, intersection = cv.intersectConvexConvex(source_hull, target_hull)
    overlap = float(area / min(source_area, target_area))

    if intersection is None or overlap < MIN_ANNOTATED_OVERLAP:
        return None

    border = ANNOTATED_BORDER_RATIO * target.scale
    source_indices = [
        i
        for i, point in enumerate(projected)
        if cv.pointPolygonTest(intersection, (float(point[0]), float(point[1])), True) >= border
    ]
    target_indices = [
        j
        for j, point in enumerate(target.points)
        if cv.pointPolygonTest(intersection, (float(point[0]), float(point[1])), True) >= border
    ]

    if min(len(source_indices), len(target_indices)) < MIN_SHARED_SPOTS:
        return None

    source_set = set(source_indices)
    target_set = set(target_indices)
    matched = sum(i in source_set and j in target_set for i, j in pairs)
    n_spots = len(source_indices) + len(target_indices)
    unmatched = 1.0 - 2.0 * matched / n_spots

    # Geometry-only absence: even a visually different nearby spot counts as present.
    distances = cdist(
        projected[source_indices],
        target.points[target_indices],
    )
    radius = UNSUPPORTED_RADIUS_RATIO * target.scale
    source_unsupported = int(np.count_nonzero(np.min(distances, axis=1) > radius))
    target_unsupported = int(np.count_nonzero(np.min(distances, axis=0) > radius))
    unsupported = (source_unsupported + target_unsupported) / n_spots

    return {
        "overlap": overlap,
        "source_spots": len(source_indices),
        "target_spots": len(target_indices),
        "matched": matched,
        "unmatched": unmatched,
        "unsupported": unsupported,
        "source_unsupported": source_unsupported,
        "target_unsupported": target_unsupported,
    }


def inspect_pair(source: SpotPhoto, target: SpotPhoto) -> dict:
    """
    Measure extra affine distortion relative to eye-mouth similarity alignment.
    """
    match = geometric_visual_match(source, target)
    baseline = initial_similarity(source.eye, source.mouth, target.eye, target.mouth)

    relative = match.matrix[:2, :2] @ np.linalg.inv(baseline[:2, :2])
    singular_values = np.linalg.svd(relative, compute_uv=False)
    anisotropy = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0
        else float("inf")
    )
    determinant = float(np.linalg.det(relative))
    scale = float(np.sqrt(abs(determinant)))

    landmarks = project_points(
        np.asarray((source.eye, source.mouth), dtype=np.float64),
        match.matrix,
    )
    expected = np.asarray((target.eye, target.mouth), dtype=np.float64)
    anchor_drift = float(max(np.linalg.norm(landmarks - expected, axis=1)) / target.scale)
    refined = not np.allclose(match.matrix, baseline)

    plausible = (
        determinant > 0
        and anisotropy <= MAX_ANISOTROPY
        and MIN_RELATIVE_SCALE <= scale <= MAX_RELATIVE_SCALE
        and anchor_drift <= MAX_ANCHOR_DRIFT
    )

    return {
        "matches": len(match.pairs),
        "refined": refined,
        "anisotropy": anisotropy,
        "scale": scale,
        "anchor_drift": anchor_drift,
        "flipped": determinant < 0,
        "plausible": plausible if refined else None,
        "missing": missing_spot_evidence(source, target, match.matrix, match.pairs),
    }


def run(project: Project) -> None:
    """
    Compare each query's strongest genuine and strongest impostor candidates.
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

    score_cache: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}
    observations: dict[str, list[tuple[float, dict]]] = {
        "genuine": [],
        "impostor": [],
    }

    print(f"Gallery: {len(gallery)} images")
    print(
        f"Provisional gate: anisotropy <= {MAX_ANISOTROPY}, "
        f"relative scale {MIN_RELATIVE_SCALE}..{MAX_RELATIVE_SCALE}, "
        f"anchor drift <= {MAX_ANCHOR_DRIFT}, no reflection"
    )
    print(
        "An unrefined eye-mouth similarity is marked 'base': "
        "its plausibility cannot be assessed by these diagnostics."
    )
    print(
        "type | query -> reference | score | spots | model | anisotropy "
        "| scale | anchor_drift | flip | gate"
    )
    print(
        "EXP-004J: annotated-spot hull overlap (not proven skin visibility); "
        f"edge margin={ANNOTATED_BORDER_RATIO:.2f}, "
        f"unsupported distance>{UNSUPPORTED_RADIUS_RATIO:.2f} x eye-mouth"
    )

    for label, photo in gallery.items():
        ranking = rank_individuals(label, photo, gallery, score_cache)
        genuine = next((item for item in ranking if item.subject_id == label[0]), None)
        impostor = next((item for item in ranking if item.subject_id != label[0]), None)

        for kind, candidate in (("genuine", genuine), ("impostor", impostor)):
            if candidate is None:
                continue

            reference = gallery[(candidate.subject_id, candidate.image_id)]
            measures = inspect_pair(photo, reference)
            observations[kind].append((candidate.score, measures))

            status = (
                "pass" if measures["plausible"] else "reject"
            ) if measures["plausible"] is not None else "n/a"
            model = "affine" if measures["refined"] else "base"

            print(
                f"{kind:8} | {label[0]}/{label[1]} -> "
                f"{candidate.subject_id}/{candidate.image_id} | "
                f"{candidate.score:.3f} | {measures['matches']:3} | {model:6} | "
                f"{measures['anisotropy']:.2f} | {measures['scale']:.2f} | "
                f"{measures['anchor_drift']:.3f} | "
                f"{'yes' if measures['flipped'] else 'no':3} | {status}"
            )

            missing = measures["missing"]

            if missing is None:
                print("  Shared annotation region: insufficient overlap or spots")
            else:
                print(
                    f"  Shared overlap={missing['overlap']:.2f}; "
                    f"spots={missing['source_spots']}+{missing['target_spots']}; "
                    f"matched={missing['matched']}; "
                    f"unmatched={missing['unmatched']:.1%}; "
                    f"unsupported={missing['unsupported']:.1%} "
                    f"({missing['source_unsupported']}+"
                    f"{missing['target_unsupported']})"
                )

    print("\nSummary (selected best reference per identity):")

    for kind, entries in observations.items():
        checked = [measures for _, measures in entries if measures["refined"]]
        passed = sum(measures["plausible"] for measures in checked)
        scores = [score for score, _ in entries]
        drifts = [measures["anchor_drift"] for measures in checked]
        anisotropies = [measures["anisotropy"] for measures in checked]

        if not entries:
            continue

        print(
            f"{kind}: {len(entries)} queries, {len(checked)} affine-refined, "
            f"{passed} pass, {len(checked) - passed} reject, "
            f"{len(entries) - len(checked)} unassessed; "
            f"median score={np.median(scores):.3f}"
        )

        if checked:
            print(
                f"  median anchor drift={np.median(drifts):.3f}; "
                f"median anisotropy={np.median(anisotropies):.2f}"
            )

    print("\nEXP-004J summary (only pairs with sufficient annotated overlap):")

    for kind, entries in observations.items():
        usable = [measures["missing"] for _, measures in entries if measures["missing"]]
        print(f"{kind}: {len(usable)}/{len(entries)} comparable pairs")

        if usable:
            print(
                f"  median unmatched={np.median([m['unmatched'] for m in usable]):.1%}; "
                f"median unsupported={np.median([m['unsupported'] for m in usable]):.1%}; "
                f"median overlap={np.median([m['overlap'] for m in usable]):.2f}"
            )

    print(
        "A missing annotation is not proof of a missing biological spot; "
        "these rates are exploratory, not an identity classifier."
    )
    print(
        "Impostor rows simulate an unknown identity: the query's entire "
        "identity is excluded from the candidate selection."
    )
    print("The gate has NOT been applied to the classifier or calibrated on held-out data.")


if __name__ == "__main__":
    run(Project.open())
