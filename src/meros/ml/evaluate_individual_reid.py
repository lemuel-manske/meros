"""
Evaluate individual identification with leave-one-image-out retrieval.

The matchers receive only image features and anatomical annotations, never
subject labels or manually verified spot correspondences. Identity labels are
used only to organize gallery entries and evaluate rankings.

Scores are fixed exploratory heuristics, not calibrated identity probabilities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from meros.ml.compare_visual_spot_matching import (
    GEOMETRIC_RADIUS_RATIO,
    SpotPhoto,
    create_encoder,
    geometric_visual_match,
    load_photo,
    visual_pairs,
)
from meros.ml.match_spot_patterns import match_spots
from meros.ml.prepare_reid_gallery import load_gallery
from meros.project import Project


TOP_K = 3
METHODS = ("proximity", "visual", "visual + geometry")


@dataclass(frozen=True)
class Candidate:
    subject_id: str
    image_id: int
    score: float


def pair_score(
    source: SpotPhoto,
    target: SpotPhoto,
    method: str,
) -> float:
    """
    Compute one fixed, normalized heuristic score for each matching method.

    Coverage is relative to the smaller set of spots, so a large count of
    annotations does not directly imply a higher score.
    """
    if method == "proximity":
        result = match_spots(
            source.points,
            target.points,
            source.eye,
            source.mouth,
            target.eye,
            target.mouth,
            radius_ratio=GEOMETRIC_RADIUS_RATIO,
        )
        pairs = result.pairs
        errors = result.errors
    elif method == "visual":
        pairs = visual_pairs(source, target)
        errors = ()
    else:
        result = geometric_visual_match(source, target)
        pairs = result.pairs
        errors = result.errors

    if not pairs:
        return 0.0

    coverage = len(pairs) / min(len(source.points), len(target.points))

    if method == "proximity":
        radius = GEOMETRIC_RADIUS_RATIO * target.scale
        qualities = [max(0.0, 1.0 - error / radius) for error in errors]
    else:
        similarities = [
            float(np.dot(source.descriptors[i], target.descriptors[j]))
            for i, j in pairs
        ]
        visual_qualities = [max(0.0, 2.0 * similarity - 1.0) for similarity in similarities]

        if method == "visual":
            qualities = visual_qualities
        else:
            radius = GEOMETRIC_RADIUS_RATIO * target.scale
            qualities = [
                visual_quality * max(0.0, 1.0 - error / radius)
                for visual_quality, error in zip(visual_qualities, errors, strict=True)
            ]

    return float(coverage * np.mean(qualities))


def rank_individuals(
    query_label: tuple[str, int],
    query: SpotPhoto,
    gallery: dict[tuple[str, int], SpotPhoto],
    method: str,
) -> list[Candidate]:
    """
    Rank distinct identities by their highest-scoring reference image.
    """
    best_by_subject: dict[str, Candidate] = {}

    for (subject_id, image_id), reference in gallery.items():
        if (subject_id, image_id) == query_label:
            continue

        score = pair_score(query, reference, method)
        previous = best_by_subject.get(subject_id)

        if previous is None or score > previous.score:
            best_by_subject[subject_id] = Candidate(subject_id, image_id, score)

    return sorted(
        best_by_subject.values(),
        key=lambda candidate: (-candidate.score, candidate.subject_id),
    )


def report_method(
    name: str,
    gallery: dict[tuple[str, int], SpotPhoto],
    queries: list[tuple[str, int]],
) -> None:
    """
    Report Rank-1, Rank-k and the true identity's rank for every query.
    """
    correct_rank1 = 0
    correct_rankk = 0
    reciprocal_ranks = []
    ties = 0

    print(f"\n{name}")
    print("query | prediction | conservative_true_rank | top candidates (score)")

    for label in queries:
        ranking = rank_individuals(label, gallery[label], gallery, name)
        true_candidate = next(
            (candidate for candidate in ranking if candidate.subject_id == label[0]),
            None,
        )

        if not ranking or true_candidate is None:
            print(f"{label[0]}/{label[1]} | no valid gallery positive")
            continue

        # Worst possible rank within a tie; alphabetical ordering earns no credit.
        true_rank = sum(
            candidate.score >= true_candidate.score - 1e-9
            for candidate in ranking
        )
        best_score = ranking[0].score
        tied_best = sum(abs(candidate.score - best_score) <= 1e-9 for candidate in ranking)
        is_tied = tied_best > 1
        ties += int(is_tied)
        correct_rank1 += int(true_rank == 1 and best_score > 0)
        correct_rankk += int(true_rank <= TOP_K and true_candidate.score > 0)
        reciprocal_ranks.append(1.0 / true_rank)

        top = ", ".join(
            f"{candidate.subject_id}/{candidate.image_id}={candidate.score:.3f}"
            for candidate in ranking[:TOP_K]
        )
        suffix = " [TIE]" if is_tied else ""
        print(
            f"{label[0]}/{label[1]} | {ranking[0].subject_id} | "
            f"{true_rank}/{len(ranking)} | {top}{suffix}"
        )

    evaluated = len(reciprocal_ranks)

    if not evaluated:
        print("No eligible queries.")
        return

    print(
        f"Rank-1 (unique winner): {correct_rank1}/{evaluated} "
        f"({correct_rank1 / evaluated:.1%})"
    )
    print(f"Rank-{TOP_K}: {correct_rankk}/{evaluated} ({correct_rankk / evaluated:.1%})")
    print(f"Conservative mean reciprocal rank: {np.mean(reciprocal_ranks):.3f}")
    print(f"Ambiguous best-score ties: {ties}/{evaluated}")


def run(project: Project) -> None:
    """
    Prepare the gallery once and evaluate all eligible queries with all methods.
    """
    encoder = create_encoder()
    annotated = load_gallery(project)
    gallery: dict[tuple[str, int], SpotPhoto] = {}

    for image in annotated:
        label = (image.subject_id, image.image_id)
        photo = load_photo(project, label, encoder)

        if photo is not None:
            gallery[label] = photo
        else:
            print(f"SKIP {label}: missing eye/mouth or insufficient spots")

    counts = Counter(label[0] for label in gallery)
    queries = [
        label
        for label in gallery
        if counts[label[0]] >= 2
    ]

    print(
        f"Gallery: {len(gallery)} images, {len(counts)} identities; "
        f"{len(queries)} leave-one-image-out queries."
    )
    print("Identity score: normalized coverage weighted by match quality.")
    print("The reference image with the best score represents each identity.")
    print("Scores are heuristics, not probabilities.")

    for method in METHODS:
        report_method(method, gallery, queries)


if __name__ == "__main__":
    run(Project.open())
