"""
Inspect whether high-scoring matches require anatomically implausible affine changes.

This is a diagnostic only. It does not change identification scores or rankings.
The provisional bounds below are not physical laws or calibrated thresholds.
"""

from __future__ import annotations

import numpy as np

from meros.ml.evaluate_individual_reid import (
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
        "Provisional gate: anisotropy <= 1.8, relative scale 0.7..1.4, "
        "anchor drift <= 0.12, no reflection"
    )
    print(
        "An unrefined eye-mouth similarity is marked 'base': "
        "its plausibility cannot be assessed by these diagnostics."
    )
    print(
        "type | query -> reference | score | spots | model | anisotropy "
        "| scale | anchor_drift | flip | gate"
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

    print(
        "Impostor rows simulate an unknown identity: the query's entire "
        "identity is excluded from the candidate selection."
    )
    print("The gate has NOT been applied to the classifier or calibrated on held-out data.")


if __name__ == "__main__":
    run(Project.open())
