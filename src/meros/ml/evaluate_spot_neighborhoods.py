"""
EXP-004N: test spot-neighborhood agreement around proposed pair matches.

The baseline identifies individual spot pairs using appearance and geometry.
Here each pair is checked against OTHER annotated spots surrounding its two
endpoints, including unmatched neighbors. This is diagnostic, not a new score.

Local comparisons are restricted to overlapping convex hulls of annotated
spots, an imperfect proxy for comparable visible skin. Sparse or occluded
annotations and longitudinal mark changes can make true pairs look different.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from meros.ml.evaluate_individual_reid import (
    MAX_VISUAL_DISTANCE,
    MIN_SPOTS,
    SpotPhoto,
    create_encoder,
    geometric_visual_match,
    load_photo,
    pair_score,
    project_points,
)
from meros.project import Project


NEIGHBOR_RADIUS = 0.40
NEIGHBOR_TOLERANCE = 0.045
ANNOTATED_BORDER = 0.03
MIN_NEIGHBORS = 3
MIN_EVALUABLE_ANCHORS = 3

# Include disputed matches and biological controls, without fitting thresholds.
INSPECT_PAIRS = (
    (("rafael", 3), ("rogerio", 1)),
    (("rafael", 3), ("rafael", 2)),
    (("rogerio", 1), ("rogerio", 4)),
    (("rogerio", 4), ("rafael", 2)),
    (("athila", 0), ("rogerio", 1)),
    (("rafael", 4), ("rogerio", 0)),
)

Label = tuple[str, int]


@dataclass(frozen=True)
class NeighborhoodResult:
    anchors: int
    matched_spots: int
    geometry: float
    geometry_visual: float
    local_count_disagreement: float


def maximum_local_matches(
    distance: np.ndarray,
    allowed: np.ndarray,
) -> int:
    """
    Find maximum one-to-one neighbor support with a dummy unmatched option.

    An allowed link always costs less than leaving both endpoints unmatched.
    Unallowed links cost more, so they cannot fabricate local support.
    """
    n_source, n_target = distance.shape
    size = n_source + n_target
    cost = np.full((size, size), 3.0, dtype=float)
    cost[:n_source, :n_target] = np.where(
        allowed, np.minimum(distance, 0.99), 3.0
    )
    cost[np.arange(n_source), n_target + np.arange(n_source)] = 0.5
    cost[n_source + np.arange(n_target), np.arange(n_target)] = 0.5
    cost[n_source:, n_target:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    return int(
        sum(
            row < n_source and col < n_target and allowed[row, col]
            for row, col in zip(rows, cols, strict=True)
        )
    )


def inspect_neighborhood(
    source: SpotPhoto,
    target: SpotPhoto,
) -> NeighborhoodResult | None:
    """
    Check local constellations surrounding automatically matched anchor spots.

    For each anchor, compare all annotated neighbors inside both shared
    footprints. The anchor position is subtracted, so disagreement is local,
    not a consequence of differences in image translation or crop placement.
    """
    match = geometric_visual_match(source, target)

    if len(match.pairs) < MIN_EVALUABLE_ANCHORS:
        return None

    projected = project_points(source.points, match.matrix)
    target_points = target.points
    source_hull = cv.convexHull(projected.astype(np.float32))
    target_hull = cv.convexHull(target_points.astype(np.float32))
    area, intersection = cv.intersectConvexConvex(source_hull, target_hull)

    if intersection is None or area <= 0:
        return None

    border = ANNOTATED_BORDER * target.scale

    def comparable(points: np.ndarray) -> set[int]:
        return {
            index
            for index, point in enumerate(points)
            if cv.pointPolygonTest(
                intersection, (float(point[0]), float(point[1])), True
            ) >= border
        }

    shared_source = comparable(projected)
    shared_target = comparable(target_points)
    radius = NEIGHBOR_RADIUS * target.scale
    tolerance = NEIGHBOR_TOLERANCE * target.scale
    geometric = []
    visual = []
    count_disagreement = []

    for source_id, target_id in match.pairs:
        if source_id not in shared_source or target_id not in shared_target:
            continue

        first_ids = [
            i for i in sorted(shared_source)
            if i != source_id and np.linalg.norm(
                projected[i] - projected[source_id]
            ) <= radius
        ]
        second_ids = [
            j for j in sorted(shared_target)
            if j != target_id and np.linalg.norm(
                target_points[j] - target_points[target_id]
            ) <= radius
        ]

        if min(len(first_ids), len(second_ids)) < MIN_NEIGHBORS:
            continue

        first = projected[first_ids] - projected[source_id]
        second = target_points[second_ids] - target_points[target_id]
        distances = cdist(first, second)
        visual_distances = cdist(
            source.descriptors[first_ids],
            target.descriptors[second_ids],
            metric="cosine",
        )
        allowed_geometry = distances <= tolerance
        allowed_visual = allowed_geometry & (visual_distances <= MAX_VISUAL_DISTANCE)
        denominator = len(first_ids) + len(second_ids)

        geometric.append(
            2 * maximum_local_matches(distances / tolerance, allowed_geometry)
            / denominator
        )
        visual.append(
            2 * maximum_local_matches(distances / tolerance, allowed_visual)
            / denominator
        )
        count_disagreement.append(
            abs(len(first_ids) - len(second_ids)) / max(len(first_ids), len(second_ids))
        )

    if len(geometric) < MIN_EVALUABLE_ANCHORS:
        return None

    return NeighborhoodResult(
        anchors=len(geometric),
        matched_spots=len(match.pairs),
        geometry=float(np.median(geometric)),
        geometry_visual=float(np.median(visual)),
        local_count_disagreement=float(np.median(count_disagreement)),
    )


def evaluate(
    photos: dict[Label, SpotPhoto],
    first: Label,
    second: Label,
) -> tuple[float, NeighborhoodResult | None]:
    """
    Return baseline pair score and independent local-neighborhood diagnostics.
    """
    source, target = photos[first], photos[second]
    return pair_score(source, target), inspect_neighborhood(source, target)


def run(project: Project) -> None:
    """
    Compare disputed pairs with verified same-identity controls.

    Do not assume these selected examples are independent training samples.
    """
    encoder = create_encoder()
    photos: dict[Label, SpotPhoto] = {}

    for subject in project.media.list_subjects():
        for image in subject.images:
            label = (subject.subject_id, image.image_id)

            if len(project.metadata.read_subject_spots(*label)) < MIN_SPOTS:
                continue

            photo = load_photo(project, label, encoder)

            if photo is not None:
                photos[label] = photo

    counts = Counter(subject for subject, _ in photos)
    known = [label for label in photos if counts[label[0]] >= 2]
    comparisons: list[tuple[Label, Label, str]] = [
        (first, second, "inspect") for first, second in INSPECT_PAIRS
    ]

    # Each known query has a strongest verified same-individual match and
    # strongest other-individual match under the original unmodified scorer.
    for first in known:
        scores = [
            (second, pair_score(photos[first], photos[second]))
            for second in photos if second != first
        ]

        for relation in ("same", "different"):
            choices = [
                (label, score) for label, score in scores
                if (label[0] == first[0]) == (relation == "same")
            ]

            if choices:
                best = max(choices, key=lambda item: item[1])
                comparisons.append((first, best[0], relation))

    print(f"Gallery: {len(photos)} photos, {len(counts)} identities")
    print(
        f"EXP-004N: neighborhood radius={NEIGHBOR_RADIUS:.2f}, "
        f"tolerance={NEIGHBOR_TOLERANCE:.3f} x target eye-mouth distance"
    )
    print(
        "pair | class | original | anchors/matches | local geometry | "
        "local geometry+visual | local count difference"
    )
    summary: dict[str, list[NeighborhoodResult]] = {
        "same": [], "different": [],
    }

    for first, second, relation in comparisons:
        baseline, result = evaluate(photos, first, second)
        pair_name = f"{first[0]}/{first[1]} -> {second[0]}/{second[1]}"
        label = "same" if first[0] == second[0] else "different"

        if result is None:
            print(f"{pair_name} | {relation} ({label}) | {baseline:.3f} | not comparable")
            continue

        print(
            f"{pair_name} | {relation} ({label}) | {baseline:.3f} | "
            f"{result.anchors}/{result.matched_spots} | "
            f"{result.geometry:.2f} | {result.geometry_visual:.2f} | "
            f"{result.local_count_disagreement:.2f}"
        )

        if relation in summary:
            summary[relation].append(result)

    for relation, results in summary.items():
        if results:
            print(
                f"{relation}: {len(results)} comparable best-match queries; "
                f"median geometry={np.median([r.geometry for r in results]):.2f}, "
                "median geometry+visual="
                f"{np.median([r.geometry_visual for r in results]):.2f}, "
                "median local count difference="
                f"{np.median([r.local_count_disagreement for r in results]):.2f}"
            )

    print(
        "No score was adjusted. Lack of local support is not proof of a "
        "different fish; compare positive and negative controls before use."
    )


if __name__ == "__main__":
    run(Project.open())
