"""
Evaluate dataset-adaptive candidate ranking from multiple spot-matching metrics.

Pair evidence is compared with verified same/different-individual examples
using a group-balanced nearest-neighbor model. Each evaluation fold excludes
the entire query identity from model fitting, but retains its other photos in
the known-identity gallery. Scores are neighbor proximity, not probabilities.

This experiment proposes candidates for human review; it never declares an
identity as verified or automatically classifies a fish as unknown.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

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
    pair_score,
    project_points,
)
from meros.project import Project


TOP_CANDIDATES = 3
MIN_SHARED_SPOTS = 5
ANNOTATED_BORDER = 0.03
GRID_SIZE = 3
FEATURE_NAMES = (
    "visual_geometric_score",
    "match_coverage",
    "unmatched_spots",
    "unsupported_spots",
    "density_log_ratio",
    "spatial_distribution",
    "annotated_overlap",
    "anchor_drift",
    "affine_anisotropy",
    "reflection",
    "comparable_region",
)


@dataclass(frozen=True)
class RankedCandidate:
    identity: str
    image_id: int
    support: float
    original_score: float


def shared_pattern_features(
    source: SpotPhoto,
    target: SpotPhoto,
    matrix: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> tuple[float, float, float, float, float, float] | None:
    """
    Measure pattern agreement only within overlapping annotated footprints.

    Footprints do not prove that every skin mark is visible or annotated.
    """
    projected = project_points(source.points, matrix).astype(np.float32)
    target_points = target.points.astype(np.float32)
    source_hull = cv.convexHull(projected)
    target_hull = cv.convexHull(target_points)
    source_area = cv.contourArea(source_hull)
    target_area = cv.contourArea(target_hull)

    if min(source_area, target_area) <= 0:
        return None

    intersection_area, intersection = cv.intersectConvexConvex(
        source_hull, target_hull
    )

    if intersection is None or intersection_area <= 0:
        return None

    overlap = float(intersection_area / min(source_area, target_area))
    border = ANNOTATED_BORDER * target.scale

    def select_shared(points: np.ndarray) -> tuple[list[int], np.ndarray]:
        indices = [
            index
            for index, point in enumerate(points)
            if cv.pointPolygonTest(
                intersection, (float(point[0]), float(point[1])), True
            ) >= border
        ]

        return indices, points[indices]

    source_ids, source_shared = select_shared(projected)
    target_ids, target_shared = select_shared(target_points)

    if min(len(source_ids), len(target_ids)) < MIN_SHARED_SPOTS:
        return None

    source_set = set(source_ids)
    target_set = set(target_ids)
    n_spots = len(source_shared) + len(target_shared)
    matches = sum(i in source_set and j in target_set for i, j in pairs)
    unmatched = 1.0 - 2.0 * matches / n_spots

    distances = cdist(source_shared, target_shared)
    radius = GEOMETRIC_RADIUS_RATIO * target.scale
    unsupported = (
        np.count_nonzero(np.min(distances, axis=1) > radius)
        + np.count_nonzero(np.min(distances, axis=0) > radius)
    ) / n_spots

    density = abs(float(np.log(len(source_shared) / len(target_shared))))
    extent = intersection.reshape(-1, 2)
    minimum = extent.min(axis=0)
    span = extent.max(axis=0) - minimum

    if np.any(span <= 0):
        return None

    def spatial_histogram(points: np.ndarray) -> np.ndarray:
        normalized = np.clip((points - minimum) / span, 0.0, 1.0 - 1e-8)
        histogram, _, _ = np.histogram2d(
            normalized[:, 0],
            normalized[:, 1],
            bins=GRID_SIZE,
            range=((0.0, 1.0), (0.0, 1.0)),
        )

        return histogram / len(points)

    distribution = float(
        0.5 * np.abs(
            spatial_histogram(source_shared) - spatial_histogram(target_shared)
        ).sum()
    )

    return unmatched, float(unsupported), density, distribution, overlap, 1.0


def pair_evidence(source: SpotPhoto, target: SpotPhoto) -> np.ndarray:
    """
    Extract a compact metric vector without altering baseline pair matching.
    """
    match = geometric_visual_match(source, target)
    baseline = initial_similarity(source.eye, source.mouth, target.eye, target.mouth)
    anchors = project_points(
        np.asarray((source.eye, source.mouth), dtype=np.float64), match.matrix
    )
    expected = np.asarray((target.eye, target.mouth), dtype=np.float64)
    drift = float(np.max(np.linalg.norm(anchors - expected, axis=1)) / target.scale)
    relative = match.matrix[:2, :2] @ np.linalg.inv(baseline[:2, :2])
    singular = np.linalg.svd(relative, compute_uv=False)
    anisotropy = float(np.log(singular[0] / max(singular[-1], 1e-9)))
    flipped = float(np.linalg.det(relative) <= 0)
    coverage = len(match.pairs) / min(len(source.points), len(target.points))

    shared = shared_pattern_features(source, target, match.matrix, match.pairs)

    if shared is None:
        shared = (np.nan, np.nan, np.nan, np.nan, np.nan, 0.0)

    return np.asarray(
        (pair_score(source, target), coverage, *shared, drift, anisotropy, flipped),
        dtype=float,
    )


class PairKNN:
    """
    Normalize and weight metrics from labeled identity-pair groups.

    One group corresponds to a positive identity or an unordered pair of
    negative identities. This prevents prolific fish from counting as dozens
    of independent identities when calculating neighbor support.
    """

    def __init__(
        self,
        samples: list[tuple[np.ndarray, tuple[str, ...]]],
    ) -> None:
        values = np.stack([vector for vector, _ in samples])
        self.center = np.array(
            [
                float(np.median(column[np.isfinite(column)]))
                if np.any(np.isfinite(column)) else 0.0
                for column in values.T
            ]
        )
        filled = np.where(np.isfinite(values), values, self.center)
        q25, q75 = np.percentile(filled, (25, 75), axis=0)
        self.scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
        self.active = q75 - q25 > 1e-6

        if not np.any(self.active):
            raise ValueError("No varying evidence features in the verified training pairs")

        normalized = np.clip((filled - self.center) / self.scale, -5.0, 5.0)

        members: dict[tuple[str, ...], list[int]] = defaultdict(list)

        for i, (_, group) in enumerate(samples):
            members[group].append(i)

        positive = [
            np.median(normalized[indices], axis=0)
            for group, indices in members.items() if group[0] == "same"
        ]
        negative = [
            np.median(normalized[indices], axis=0)
            for group, indices in members.items() if group[0] == "diff"
        ]

        if not positive or not negative:
            raise ValueError("Both verified same- and different-identity pairs are needed")

        separation = np.abs(
            np.median(positive, axis=0) - np.median(negative, axis=0)
        )
        # Conservative shrinkage; weak features are down-weighted, not deleted.
        self.weights = np.where(
            self.active, 0.5 + 0.5 * np.minimum(separation, 2.0), 0.0
        )
        self.weights /= max(float(np.mean(self.weights[self.active])), 1e-9)

        self.samples = normalized * np.sqrt(self.weights)
        self.groups = list(members.items())

    def support(self, vector: np.ndarray) -> float:
        """
        Contrast typical nearest same-identity and different-identity groups.

        Support is relative proximity in feature space, NOT a probability.
        """
        filled = np.where(np.isfinite(vector), vector, self.center)
        normalized = np.clip((filled - self.center) / self.scale, -5.0, 5.0)
        point = normalized * np.sqrt(self.weights)
        distances = np.linalg.norm(self.samples - point, axis=1)
        positive = []
        negative = []

        for group, indices in self.groups:
            nearest = float(np.min(distances[indices]))
            (positive if group[0] == "same" else negative).append(nearest)

        def neighborhood(group_distances: list[float]) -> float:
            k = max(1, round(len(group_distances) ** 0.5))
            return float(np.mean(sorted(group_distances)[:k]))

        near_same = neighborhood(positive)
        near_diff = neighborhood(negative)

        if near_same + near_diff < 1e-12:
            return 0.5

        return near_diff / (near_same + near_diff)


def rank_candidates(
    label: tuple[str, int],
    gallery: dict[tuple[str, int], SpotPhoto],
    evidence_cache: dict[tuple[tuple[str, int], tuple[str, int]], np.ndarray],
    model: PairKNN,
    *,
    unknown: bool,
) -> list[RankedCandidate]:
    """
    Return one best photograph per identity using learned pair evidence.
    """
    candidates: dict[str, RankedCandidate] = {}

    for reference_label in gallery:
        if reference_label == label or (unknown and reference_label[0] == label[0]):
            continue

        pair = (label, reference_label)
        evidence = evidence_cache[pair]
        item = RankedCandidate(
            reference_label[0], reference_label[1], model.support(evidence),
            float(evidence[0]),
        )
        previous = candidates.get(item.identity)

        if previous is None or (item.support, item.original_score) > (
            previous.support, previous.original_score
        ):
            candidates[item.identity] = item

    return sorted(
        candidates.values(),
        key=lambda item: (-item.support, -item.original_score, item.identity),
    )


def run(project: Project) -> None:
    """
    Evaluate identity-held-out metric learning with human candidate shortlists.
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
    labels = list(gallery)
    identity_names = sorted(counts)
    evidence_cache = {
        (first, second): pair_evidence(gallery[first], gallery[second])
        for first in labels for second in labels if first != second
    }
    known_total = 0
    known_top1 = 0
    known_topk = 0
    unknown_total = 0
    per_fold_weights = []

    print(f"Gallery: {len(gallery)} images, {len(counts)} identities")
    print("Pair KNN: adaptive scales and feature weights from verified identities.")
    print(
        "Each fold excludes the query identity from metric training. "
        "Candidates are for REVIEW, not automatic identification."
    )
    print("query | mode | candidates = neighbor-support [original score]")

    for heldout in identity_names:
        training = []

        for first in labels:
            if first[0] == heldout:
                continue

            for second in labels:
                if second == first or second[0] == heldout:
                    continue

                same = first[0] == second[0]
                group = (
                    ("same", first[0])
                    if same else ("diff", *sorted((first[0], second[0])))
                )
                training.append((evidence_cache[(first, second)], group))

        try:
            model = PairKNN(training)
        except ValueError:
            print(f"{heldout}: insufficient verified training groups; REVIEW")
            continue

        per_fold_weights.append(model.weights)

        for label in labels:
            if label[0] != heldout:
                continue

            for unknown in (False, True):
                if not unknown and counts[heldout] < 2:
                    continue

                ranking = rank_candidates(
                    label, gallery, evidence_cache, model, unknown=unknown
                )
                mode = "unknown" if unknown else "known"
                shortlist = ", ".join(
                    f"{item.identity}/{item.image_id}="
                    f"{item.support:.3f}[{item.original_score:.3f}]"
                    for item in ranking[:TOP_CANDIDATES]
                )
                print(f"{label[0]}/{label[1]} | {mode} | {shortlist or 'none'}")

                if unknown:
                    unknown_total += 1
                elif ranking:
                    known_total += 1
                    actual = next(
                        (i for i, item in enumerate(ranking) if item.identity == heldout),
                        None,
                    )
                    known_top1 += int(actual == 0)
                    known_topk += int(actual is not None and actual < TOP_CANDIDATES)

    print(
        f"Known identity-held-out: Rank-1={known_top1}/{known_total}, "
        f"Rank-{TOP_CANDIDATES}={known_topk}/{known_total}"
    )
    print(
        f"Unknown identity-held-out: {unknown_total} queries, all require "
        "human review (no calibrated acceptance threshold)."
    )

    if per_fold_weights:
        importance = np.median(per_fold_weights, axis=0)
        print("Median adaptive feature weights across held-out folds:")

        for name, weight in sorted(
            zip(FEATURE_NAMES, importance, strict=True),
            key=lambda item: -item[1],
        ):
            print(f"  {name}: {weight:.2f}")

    print(
        "These relative neighbor scores are not identity probabilities. "
        "Independent encounters and more confirmed identities are needed "
        "before calibrating AUTO versus REVIEW."
    )


if __name__ == "__main__":
    run(Project.open())
