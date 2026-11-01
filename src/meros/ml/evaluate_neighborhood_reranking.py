"""
EXP-004O: evaluate neighborhood-penalized matching on every ordered photo pair.

The baseline remains unchanged. Compare simple fixed penalties derived from
EXP-004N's local-neighborhood consistency, without fitting any weights.
When neighborhoods are not comparable, leave the original score unchanged,
mark the missing evidence, and report how often that fallback matters.

All identity queries are evaluated against every other image. Known queries
have another photo of the same individual; simulated unknown queries exclude
every photo of the query identity. Scores are not identity probabilities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from meros.ml.evaluate_individual_reid import (
    MIN_SPOTS,
    SpotPhoto,
    create_encoder,
    load_photo,
    pair_score,
)
from meros.ml.evaluate_spot_neighborhoods import NeighborhoodResult, inspect_neighborhood
from meros.project import Project


TOP_K = 3
METHODS = ("baseline", "geometry", "visual", "soft_visual")

Label = tuple[str, int]
Pair = tuple[Label, Label]


@dataclass(frozen=True)
class Evidence:
    original: float
    neighborhood: NeighborhoodResult | None

    def score(self, method: str) -> float:
        """
        Penalize disagreement only where neighborhood evidence is comparable.
        """
        if method == "baseline" or self.neighborhood is None:
            return self.original

        if method == "geometry":
            factor = self.neighborhood.geometry
        else:
            factor = self.neighborhood.geometry_visual

        if method == "soft_visual":
            factor = 0.5 + 0.5 * factor

        return self.original * factor


@dataclass(frozen=True)
class Candidate:
    subject_id: str
    image_id: int
    score: float
    comparable: bool


def candidates(
    query: Label,
    labels: list[Label],
    evidence: dict[Pair, Evidence],
    method: str,
    *,
    unknown: bool = False,
) -> list[Candidate]:
    """
    Select each identity's best reference after rescoring all photo pairs.
    """
    best: dict[str, Candidate] = {}

    for reference in labels:
        if query == reference or (unknown and query[0] == reference[0]):
            continue

        item = evidence[(query, reference)]
        candidate = Candidate(
            reference[0], reference[1], item.score(method),
            item.neighborhood is not None,
        )
        current = best.get(reference[0])

        if current is None or candidate.score > current.score:
            best[reference[0]] = candidate

    return sorted(best.values(), key=lambda item: (-item.score, item.subject_id))


def correct_rank(ranking: list[Candidate], identity: str) -> int | None:
    """
    Rank the true identity conservatively, putting equal scores ahead of it.
    """
    correct = next((item for item in ranking if item.subject_id == identity), None)

    if correct is None or correct.score <= 0:
        return None

    return sum(item.score >= correct.score - 1e-9 for item in ranking)


def display(item: Candidate | None) -> str:
    """
    Print an identity's reference, score and neighborhood comparability.
    """
    if item is None:
        return "none"

    flag = "" if item.comparable else " [no-neighborhood]"
    return f"{item.subject_id}/{item.image_id}={item.score:.3f}{flag}"


def summarize_pairs(
    labels: list[Label],
    evidence: dict[Pair, Evidence],
) -> None:
    """
    Report evidence availability for all directed genuine and impostor pairs.
    """
    for same in (True, False):
        pairs = [
            item for (first, second), item in evidence.items()
            if (first[0] == second[0]) == same
        ]
        comparable = [item.neighborhood for item in pairs if item.neighborhood is not None]
        name = "same" if same else "different"

        if comparable:
            geom = [item.geometry for item in comparable]
            visual = [item.geometry_visual for item in comparable]
            print(
                f"{name} ordered pairs: {len(pairs)}, comparable={len(comparable)}, "
                f"missing={len(pairs) - len(comparable)}; "
                f"median geometry={np.median(geom):.3f}, "
                f"median visual={np.median(visual):.3f}"
            )
        else:
            print(f"{name} ordered pairs: {len(pairs)}, no comparable neighborhoods")


def run(project: Project) -> None:
    """
    Compare fixed neighborhood penalties and unchanged original matcher.
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
    counts = Counter(identity for identity, _ in labels)
    evidence: dict[Pair, Evidence] = {}

    print(f"Gallery: {len(labels)} photos, {len(counts)} identities")
    print("EXP-004O: compare baseline, original*geometry, original*visual,")
    print("and original*(0.5+0.5*visual) on ALL directed photo pairs.")
    print("No neighborhood: neutral factor=1, explicitly reported as missing.")
    print("No parameter learned or automatic identity/unknown decision.")

    for first in labels:
        for second in labels:
            if first == second:
                continue

            source, target = gallery[first], gallery[second]
            original = pair_score(source, target)

            if original > 0:
                neighborhood = inspect_neighborhood(source, target)
            else:
                neighborhood = None

            evidence[(first, second)] = Evidence(original, neighborhood)

    summarize_pairs(labels, evidence)

    known = [label for label in labels if counts[label[0]] > 1]
    ranks: dict[str, list[int | None]] = {method: [] for method in METHODS}
    margins: dict[str, list[float]] = {method: [] for method in METHODS}
    fallback_winners = Counter()

    print("\nKnown queries: ranking by best photo of each identity")
    print("query | method | correct rank | first candidate | best impostor | margin")

    for label in known:
        for method in METHODS:
            ranking = candidates(label, labels, evidence, method)
            true_rank = correct_rank(ranking, label[0])
            ranks[method].append(true_rank)
            best_true = next(item for item in ranking if item.subject_id == label[0])
            best_wrong = next(item for item in ranking if item.subject_id != label[0])
            margin = best_true.score - best_wrong.score
            margins[method].append(margin)

            if ranking and not ranking[0].comparable:
                fallback_winners[method] += 1

            rank = str(true_rank) if true_rank is not None else "none"
            print(
                f"{label[0]}/{label[1]} | {method} | {rank} | "
                f"{display(ranking[0])} | {display(best_wrong)} | {margin:+.3f}"
            )

    print("\nKnown-summary (conservative tie handling)")
    print("method | rank-1 | rank-3 | median true-vs-impostor margin | missing top")
    for method in METHODS:
        evaluated = ranks[method]
        top1 = sum(rank == 1 for rank in evaluated)
        topk = sum(rank is not None and rank <= TOP_K for rank in evaluated)
        print(
            f"{method} | {top1}/{len(evaluated)} | {topk}/{len(evaluated)} | "
            f"{np.median(margins[method]):+.3f} | {fallback_winners[method]}"
        )

    print("\nPer-query changes in correct identity rank compared with baseline")
    baseline = ranks["baseline"]
    for method in METHODS[1:]:
        improved = []
        degraded = []

        for label, before, after in zip(known, baseline, ranks[method], strict=True):
            first_rank = before if before is not None else len(counts) + 1
            second_rank = after if after is not None else len(counts) + 1

            if second_rank < first_rank:
                improved.append(f"{label[0]}/{label[1]}")
            elif second_rank > first_rank:
                degraded.append(f"{label[0]}/{label[1]}")

        print(
            f"{method}: improved={improved or 'none'}; "
            f"degraded={degraded or 'none'}"
        )

    print("\nSimulated unknown: query identity entirely absent from the gallery")
    print("method | queries | median best score | top matches lacking neighborhood")

    for method in METHODS:
        best = [
            candidates(label, labels, evidence, method, unknown=True)[0]
            for label in labels if len(counts) > 1
        ]
        missing = sum(not candidate.comparable for candidate in best)
        print(
            f"{method} | {len(best)} | "
            f"{np.median([candidate.score for candidate in best]):.3f} | "
            f"{missing}"
        )

    print(
        "No unknown-acceptance threshold is calibrated. These pairs share "
        "individual identities and encounters, so the counts above are not "
        "independent evidence of generalization."
    )


if __name__ == "__main__":
    run(Project.open())
