"""
EXP-004M: rerank plausible individual candidates with combined spot evidence.

Retrieve candidates with the unchanged visual/geometric matcher, require
minimal local support relative to the strongest match, and compare feature
ablations using an identity-held-out pair KNN. Scores are not probabilities.

This is only a shortlist for human review, never automatic identification.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from meros.ml.evaluate_adaptive_candidates import FEATURE_NAMES, PairKNN, pair_evidence
from meros.ml.evaluate_individual_reid import (
    MIN_SPOTS,
    Candidate,
    SpotPhoto,
    create_encoder,
    load_photo,
    rank_individuals,
)
from meros.project import Project


TOP_CANDIDATES = 3
MIN_MATCHED_SPOTS = 3
MIN_RELATIVE_SCORE = 0.50

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "local": (
        "visual_geometric_score",
        "match_coverage",
        "unmatched_spots",
        "unsupported_spots",
    ),
    "global": (
        "visual_geometric_score",
        "density_log_ratio",
        "spatial_distribution",
    ),
    "full": FEATURE_NAMES,
}
METHODS = ("baseline", "retrieval", *FEATURE_SETS)


def training_pairs(
    labels: list[tuple[str, int]],
    heldout: str,
    evidence: dict[tuple[tuple[str, int], tuple[str, int]], np.ndarray],
) -> list[tuple[np.ndarray, tuple[str, ...]]]:
    """
    Fit only on comparisons involving identities other than the query fish.
    """
    samples = []

    for first in labels:
        if first[0] == heldout:
            continue

        for second in labels:
            if second == first or second[0] == heldout:
                continue

            group = (
                ("same", first[0])
                if first[0] == second[0]
                else ("diff", *sorted((first[0], second[0])))
            )
            samples.append((evidence[(first, second)], group))

    return samples


def retrieve(
    query_label: tuple[str, int],
    ranking: list[Candidate],
    gallery: dict[tuple[str, int], SpotPhoto],
    evidence: dict[tuple[tuple[str, int], tuple[str, int]], np.ndarray],
) -> list[Candidate]:
    """
    Only pass original top candidates with meaningful local matching evidence.

    The ratio is query-relative, not an identity-classification threshold.
    Three matched spots is a provisional minimum for supported geometry.
    """
    if not ranking or ranking[0].score <= 0:
        return []

    strongest = ranking[0].score
    selected = []

    for candidate in ranking:
        if len(selected) >= TOP_CANDIDATES:
            break

        if candidate.score < strongest * MIN_RELATIVE_SCORE:
            continue

        reference_label = (candidate.subject_id, candidate.image_id)
        coverage = float(evidence[(query_label, reference_label)][1])
        min_points = min(
            len(gallery[query_label].points),
            len(gallery[reference_label].points),
        )
        matched_spots = round(coverage * min_points)

        if matched_spots < MIN_MATCHED_SPOTS:
            continue

        selected.append(candidate)

    return selected


def rerank(
    label: tuple[str, int],
    selected: list[Candidate],
    evidence: dict[tuple[tuple[str, int], tuple[str, int]], np.ndarray],
    model: PairKNN,
) -> list[tuple[Candidate, float]]:
    """
    Rerank retrieved identity representatives without promoting excluded pairs.
    """
    result = [
        (
            candidate,
            model.support(evidence[(label, (candidate.subject_id, candidate.image_id))]),
        )
        for candidate in selected
    ]

    return sorted(
        result,
        key=lambda item: (-item[1], -item[0].score, item[0].subject_id),
    )


def describe(ranking: list[tuple[Candidate, float]]) -> str:
    """
    Format first candidates with neighbor proximity and original local score.
    """
    if not ranking:
        return "none"

    return ", ".join(
        f"{candidate.subject_id}/{candidate.image_id}="
        f"{support:.3f}[{candidate.score:.3f}]"
        for candidate, support in ranking[:TOP_CANDIDATES]
    )


def run(project: Project) -> None:
    """
    Compare original, retrieved and adaptive rankings on matched query folds.
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

    labels = sorted(gallery)
    counts = Counter(label[0] for label in labels)
    evidence = {
        (first, second): pair_evidence(gallery[first], gallery[second])
        for first in labels for second in labels if first != second
    }
    score_cache = {pair: float(vector[0]) for pair, vector in evidence.items()}
    rank1 = Counter()
    rankk = Counter()
    retrieved_known = 0
    known_total = 0
    unknown_total = 0
    full_weights = []

    print(f"Gallery: {len(gallery)} images, {len(counts)} identities")
    print(
        f"EXP-004M: top {TOP_CANDIDATES} original candidates with >= "
        f"{MIN_MATCHED_SPOTS} matched spots and >= {MIN_RELATIVE_SCORE:.0%} "
        "of the query's best original score."
    )
    print(
        "Each identity is excluded from kNN training in its own fold. "
        "All results are proposals for human REVIEW."
    )

    for heldout in sorted(counts):
        samples = training_pairs(labels, heldout, evidence)
        positive_groups = {
            group for _, group in samples if group[0] == "same"
        }

        try:
            models = {
                name: PairKNN(samples, selected_features=features)
                for name, features in FEATURE_SETS.items()
            }
        except ValueError as error:
            print(f"{heldout}: no usable training model ({error})")
            continue

        full_weights.append(models["full"].weights)
        print(
            f"\nFold {heldout}: {len(positive_groups)} independent "
            "positive training identities"
        )

        for label in labels:
            if label[0] != heldout:
                continue

            for unknown in (False, True):
                if not unknown and counts[heldout] < 2:
                    continue

                original = rank_individuals(
                    label,
                    gallery[label],
                    gallery,
                    score_cache,
                    excluded_subject=heldout if unknown else None,
                )
                selected = retrieve(label, original, gallery, evidence)
                mode = "unknown" if unknown else "known"
                original_top = (
                    f"{original[0].subject_id}/{original[0].image_id}="
                    f"{original[0].score:.3f}" if original else "none"
                )
                print(
                    f"{label[0]}/{label[1]} | {mode} | "
                    f"original={original_top} | eligible={len(selected)}/{len(original)}"
                )

                results = {
                    "baseline": [(candidate, candidate.score) for candidate in original],
                    "retrieval": [(candidate, candidate.score) for candidate in selected],
                }

                for name, model in models.items():
                    results[name] = rerank(label, selected, evidence, model)

                for name, ranking in results.items():
                    if name in ("baseline", "retrieval"):
                        shortlist = ", ".join(
                            f"{candidate.subject_id}/{candidate.image_id}="
                            f"{candidate.score:.3f}"
                            for candidate, _ in ranking[:TOP_CANDIDATES]
                        ) or "none"
                    else:
                        shortlist = describe(ranking)

                    print(f"  {name}: {shortlist}")

                    if not unknown:
                        true_rank = next(
                            (
                                i
                                for i, (candidate, _) in enumerate(ranking)
                                if candidate.subject_id == heldout
                            ),
                            None,
                        )
                        rank1[name] += int(true_rank == 0)
                        rankk[name] += int(
                            true_rank is not None and true_rank < TOP_CANDIDATES
                        )

                if unknown:
                    unknown_total += 1
                else:
                    known_total += 1
                    retrieved_known += int(
                        any(candidate.subject_id == heldout for candidate in selected)
                    )

    print(f"\nKnown queries: {known_total}; retained genuine: {retrieved_known}")
    print(f"Method | Rank-1 | Rank-{TOP_CANDIDATES}")

    for name in METHODS:
        print(
            f"{name} | {rank1[name]}/{known_total} | "
            f"{rankk[name]}/{known_total}"
        )

    print(
        f"Simulated unknown queries: {unknown_total}; all are REVIEW. "
        "No open-set decision threshold is evaluated."
    )

    if full_weights:
        print("Full-model median evidence weights (held-out folds):")
        importance = np.median(full_weights, axis=0)

        for name, weight in sorted(
            zip(FEATURE_NAMES, importance, strict=True),
            key=lambda item: -item[1],
        ):
            print(f"  {name}: {weight:.2f}")

    print(
        "Feature selection, shortlist size and retrieval ratio are exploratory. "
        "With only two repeat-photographed identities, this experiment cannot "
        "validate generalization to unseen individuals."
    )


if __name__ == "__main__":
    run(Project.open())
