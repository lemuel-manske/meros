"""
Evaluate automatic re-identification with abstention for uncertain cases.

This diagnostic reuses the unchanged visual-geometric matcher. The score and
margin thresholds were chosen on explored data; they are not calibrated.
REVIEW means human verification is required, not that a fish is new.
"""

from __future__ import annotations

from collections import Counter

from meros.ml.evaluate_individual_reid import (
    MIN_SPOTS,
    Candidate,
    SpotPhoto,
    create_encoder,
    load_photo,
    rank_individuals,
)
from meros.project import Project


AUTO_MIN_SCORE = 0.22
AUTO_MIN_MARGIN = 0.08


def decision(ranking: list[Candidate]) -> str:
    """
    Return AUTO only for a sufficiently strong and separated first candidate.
    """
    if len(ranking) < 2:
        return "REVIEW"

    if ranking[0].score < AUTO_MIN_SCORE:
        return "REVIEW"

    if ranking[0].score - ranking[1].score < AUTO_MIN_MARGIN:
        return "REVIEW"

    return "AUTO"


def evaluate(
    gallery: dict[tuple[str, int], SpotPhoto],
    cache: dict[tuple[tuple[str, int], tuple[str, int]], float],
    labels: list[tuple[str, int]],
    *,
    known: bool,
) -> None:
    """
    Compare decisions for known queries and leave-one-identity-out unknowns.
    """
    accepted = 0
    correct_accepted = 0
    correct_baseline = 0
    evaluated = 0

    print(f"\n{'KNOWN' if known else 'UNKNOWN'}: {len(labels)} queries")
    print("query | best | score | margin | decision")

    for label in labels:
        ranking = rank_individuals(
            label,
            gallery[label],
            gallery,
            cache,
            excluded_subject=None if known else label[0],
        )

        if not ranking:
            print(f"{label[0]}/{label[1]} | no candidate | REVIEW")
            continue

        evaluated += 1
        best = ranking[0]
        margin = best.score - ranking[1].score if len(ranking) > 1 else 0.0
        status = decision(ranking)
        unique = sum(abs(item.score - best.score) <= 1e-9 for item in ranking) == 1
        correct = known and unique and best.score > 0 and best.subject_id == label[0]
        correct_baseline += int(correct)

        if status == "AUTO":
            accepted += 1
            correct_accepted += int(correct)

        print(
            f"{label[0]}/{label[1]} | {best.subject_id}/{best.image_id} | "
            f"{best.score:.3f} | {margin:.3f} | {status}"
        )

    if known:
        print(
            f"Known: Rank-1={correct_baseline}/{evaluated}; "
            f"AUTO={accepted}/{evaluated}; "
            f"correct AUTO={correct_accepted}/{accepted}; "
            f"wrong AUTO={accepted - correct_accepted}; "
            f"REVIEW={len(labels) - accepted}"
        )
    else:
        print(
            f"Unknown: false AUTO={accepted}/{evaluated}; "
            f"REVIEW={len(labels) - accepted}/{len(labels)}"
        )


def run(project: Project) -> None:
    """
    Evaluate fixed score-margin abstention using Project image annotations.
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

    counts = Counter(label[0] for label in gallery)
    known_queries = [label for label in gallery if counts[label[0]] >= 2]
    cache: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}

    print(f"Gallery: {len(gallery)} images, {len(counts)} identities")
    print(
        f"Selective triage: min_score={AUTO_MIN_SCORE:.2f}, "
        f"min_margin={AUTO_MIN_MARGIN:.2f}; thresholds are not validated."
    )

    evaluate(gallery, cache, known_queries, known=True)
    evaluate(gallery, cache, list(gallery), known=False)

    print(
        "UNKNOWN means the true identity was entirely removed from the gallery. "
        "AUTO is provisional; REVIEW is not an unknown-individual prediction."
    )


if __name__ == "__main__":
    run(Project.open())
