"""
Compare the original spot matcher with a local geometric-neighborhood penalty.

EXP-004N/O found that original score times neighborhood geometry preserved
14/14 known rankings while increasing the median correct-versus-impostor
margin. This file unifies their diagnostics, without learned weights.

Shared annotation hulls only approximate comparable visible skin. Sparse
annotations, viewing angle and changing mark appearance can affect the score.
Missing neighborhood evidence is neutral and cannot establish identity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from meros.ml.evaluate_individual_reid import (
    MIN_SPOTS,
    Candidate,
    SpotPhoto,
    create_encoder,
    geometric_visual_match,
    load_photo,
    pair_score,
    project_points,
    rank_individuals,
)
from meros.project import Project


NEIGHBOR_RADIUS = 0.40
NEIGHBOR_TOLERANCE = 0.045
ANNOTATED_BORDER = 0.03
MIN_NEIGHBORS = 3
MIN_EVALUABLE_ANCHORS = 3
TOP_K = 3

# Illustrative same- and different-individual pairs from the visual audit.
INSPECT_PAIRS = (
    (("rafael", 3), ("rogerio", 1)),
    (("rafael", 3), ("rafael", 2)),
    (("rogerio", 1), ("rogerio", 4)),
    (("athila", 0), ("rogerio", 1)),
)

Label = tuple[str, int]


@dataclass(frozen=True)
class NeighborhoodResult:
    anchors: int
    matched_spots: int
    geometry: float
    local_count_disagreement: float


def maximum_local_matches(
    distance: np.ndarray,
    allowed: np.ndarray,
) -> int:
    """
    Measure one-to-one local support with a minimum-cost assignment.

    Pairs outside the geometric tolerance cost more than unmatched options.
    Keep this assignment unchanged to preserve EXP-004N/O comparability.
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
        allowed_geometry = distances <= tolerance
        denominator = len(first_ids) + len(second_ids)

        geometric.append(
            2 * maximum_local_matches(distances / tolerance, allowed_geometry)
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
        local_count_disagreement=float(np.median(count_disagreement)),
    )


@dataclass(frozen=True)
class PairEvidence:
    original: float
    neighborhood: NeighborhoodResult | None

    @property
    def combined(self) -> float:
        """
        Preserve the baseline score when no neighborhood can be compared.
        """
        if self.neighborhood is None:
            return self.original

        return self.original * self.neighborhood.geometry


def correct_rank(ranking: list[Candidate], subject_id: str) -> int | None:
    """
    Worst-case true identity rank among ties; zero scores mean no evidence.
    """
    genuine = next((candidate for candidate in ranking if candidate.subject_id == subject_id), None)

    if genuine is None or genuine.score <= 0:
        return None

    return sum(candidate.score >= genuine.score - 1e-9 for candidate in ranking)


def describe_pair(pair: tuple[Label, Label], item: PairEvidence) -> str:
    """
    Display one pair's original score and comparable local geometry.
    """
    first, second = pair
    heading = f"{first[0]}/{first[1]} -> {second[0]}/{second[1]}"

    if item.neighborhood is None:
        return f"{heading} | score={item.original:.3f} | neighborhood=n/a"

    result = item.neighborhood
    return (
        f"{heading} | score={item.original:.3f} | "
        f"local={result.geometry:.3f} ({result.anchors}/{result.matched_spots} anchors) "
        f"| neighbor count difference={result.local_count_disagreement:.3f}"
    )


def run(project: Project) -> None:
    """
    Evaluate all ordered photo pairs and baseline versus neighborhood rankings.
    """
    encoder = create_encoder()
    gallery: dict[Label, SpotPhoto] = {}

    for subject in project.media.list_subjects():
        for image in subject.images:
            label = (subject.subject_id, image.image_id)

            if len(project.metadata.read_subject_spots(*label)) < MIN_SPOTS:
                continue

            photo = load_photo(project, label, encoder)

            if photo is not None:
                gallery[label] = photo

    labels = sorted(gallery)
    counts = Counter(subject for subject, _ in labels)
    evidence: dict[tuple[Label, Label], PairEvidence] = {}

    for first in labels:
        for second in labels:
            if first == second:
                continue

            original = pair_score(gallery[first], gallery[second])
            local = (
                inspect_neighborhood(gallery[first], gallery[second])
                if original > 0 else None
            )
            evidence[(first, second)] = PairEvidence(original, local)

    baseline = {pair: item.original for pair, item in evidence.items()}
    combined = {pair: item.combined for pair, item in evidence.items()}

    print(f"Gallery: {len(labels)} photos, {len(counts)} identities")
    print("Baseline: visual/geometric spot matching.")
    print("Neighborhood: baseline * geometric local consistency; no learning.")
    print("No comparable neighborhood: unchanged baseline score, not a negative match.")

    for same in (True, False):
        group = [
            item for (first, second), item in evidence.items()
            if (first[0] == second[0]) == same
        ]
        comparable = [
            item.neighborhood for item in group if item.neighborhood is not None
        ]
        name = "same" if same else "different"
        median = (
            f"{np.median([item.geometry for item in comparable]):.3f}"
            if comparable else "n/a"
        )
        print(
            f"{name} ordered pairs: {len(group)}, comparable={len(comparable)}, "
            f"missing={len(group) - len(comparable)}, median local geometry={median}"
        )

    print("\nAudited examples:")
    for pair in INSPECT_PAIRS:
        if pair in evidence:
            print("  " + describe_pair(pair, evidence[pair]))

    known = [label for label in labels if counts[label[0]] > 1]
    results: dict[str, dict[Label, tuple[int | None, float, bool]]] = {
        "baseline": {}, "neighborhood": {},
    }

    print("\nKnown queries: predicted first / best wrong / true-vs-wrong margin")
    for label in known:
        for method, scores in (("baseline", baseline), ("neighborhood", combined)):
            ranking = rank_individuals(label, gallery[label], gallery, scores)
            genuine = next(item for item in ranking if item.subject_id == label[0])
            impostor = next(item for item in ranking if item.subject_id != label[0])
            rank = correct_rank(ranking, label[0])
            margin = genuine.score - impostor.score
            top = ranking[0]
            comparable = evidence[(label, (top.subject_id, top.image_id))].neighborhood
            missing_top = comparable is None
            results[method][label] = (rank, margin, missing_top)
            flag = " [no-neighborhood]" if missing_top else ""

            print(
                f"{label[0]}/{label[1]} | {method} | rank={rank or 'n/a'} | "
                f"{top.subject_id}/{top.image_id}={top.score:.3f}{flag} | "
                f"{impostor.subject_id}/{impostor.image_id}={impostor.score:.3f} "
                f"| margin={margin:+.3f}"
            )

    print("\nMethod | Rank-1 | Rank-3 | median margin | top without neighborhood")
    for method, values in results.items():
        rows = list(values.values())
        rank1 = sum(rank == 1 for rank, _, _ in rows)
        rankk = sum(rank is not None and rank <= TOP_K for rank, _, _ in rows)
        margins = [margin for _, margin, _ in rows]
        missing_top = sum(missing for _, _, missing in rows)
        print(
            f"{method} | {rank1}/{len(rows)} | {rankk}/{len(rows)} | "
            f"{np.median(margins):+.3f} | {missing_top}"
        )

    gains = []
    losses = []

    for label in known:
        before = results["baseline"][label]
        after = results["neighborhood"][label]
        if after[1] > before[1] + 1e-9:
            gains.append(label)
        elif after[1] < before[1] - 1e-9:
            losses.append(label)

    def show_labels(items: list[Label]) -> str:
        return ", ".join(f"{subject}/{image}" for subject, image in items) or "none"

    print(f"Margin increased ({len(gains)}): {show_labels(gains)}")
    print(f"Margin decreased ({len(losses)}): {show_labels(losses)}")

    print("\nSimulated unknown (query identity absent from gallery):")
    for method, scores in (("baseline", baseline), ("neighborhood", combined)):
        top = [
            rank_individuals(
                label, gallery[label], gallery, scores, excluded_subject=label[0]
            )[0]
            for label in labels if len(counts) > 1
        ]
        missing = sum(
            evidence[(label, (candidate.subject_id, candidate.image_id))].neighborhood
            is None
            for label, candidate in zip(labels, top, strict=True)
        )
        print(
            f"{method}: {len(top)} queries, median best incorrect score="
            f"{np.median([candidate.score for candidate in top]):.3f}, "
            f"top without neighborhood={missing}"
        )

    print(
        "All scores are heuristics. Repeated photos of just two individuals "
        "limit generalization; no automatic unknown threshold is validated."
    )


if __name__ == "__main__":
    run(Project.open())
