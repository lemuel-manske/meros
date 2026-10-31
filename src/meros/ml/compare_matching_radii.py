"""
Measure how the matching radius changes geometric correspondence quality.

Known manual spot-to-spot correspondences are used strictly for evaluation,
after each matcher run. Different-individual comparisons provide impostor
counts, not verified correspondence accuracy.
"""

from __future__ import annotations

import numpy as np

from meros.ml.match_spot_patterns import match_spots, project_points, verified_pairs
from meros.project import Project


RADIUS_RATIOS = (0.03, 0.05, 0.08, 0.12)
SAME_INDIVIDUAL_PAIRS = (
    ("luana", 0, 1),
    ("rafael", 0, 1),
    ("rogerio", 0, 1),
    ("rogerio", 0, 9),
    ("yam", 0, 1),
)
DIFFERENT_INDIVIDUAL_PAIRS = (
    (("luana", 0), ("rafael", 1)),
)


def evaluate_pair(
    project: Project,
    source_label: tuple[str, int],
    target_label: tuple[str, int],
) -> None:
    """
    Evaluate each radius on a fixed pair of manually annotated photographs.
    """
    source_eye = project.metadata.read_subject_eye(*source_label)
    target_eye = project.metadata.read_subject_eye(*target_label)
    source_mouth = project.metadata.read_subject_mouth(*source_label)
    target_mouth = project.metadata.read_subject_mouth(*target_label)

    if (
        source_eye is None
        or target_eye is None
        or source_mouth is None
        or target_mouth is None
    ):
        print(f"SKIP {source_label} vs {target_label}: missing eye or mouth")
        return

    source = np.asarray(project.metadata.read_subject_spots(*source_label), dtype=float)
    target = np.asarray(project.metadata.read_subject_spots(*target_label), dtype=float)

    if len(source) < 3 or len(target) < 3:
        print(f"SKIP {source_label} vs {target_label}: insufficient spots")
        return

    same_individual = source_label[0] == target_label[0]
    references = (
        verified_pairs(project, source_label, target_label, source, target)
        if same_individual
        else set()
    )

    if same_individual and not references:
        print(f"SKIP {source_label} vs {target_label}: no indexed verified pairs")
        return

    eye_mouth_scale = float(
        np.linalg.norm(np.asarray(target_mouth) - np.asarray(target_eye.center))
    )

    print(
        f"\n{source_label} vs {target_label} "
        f"({'same' if same_individual else 'different'} individual)"
    )
    print(
        "ratio | radius_px | proposed | verified | conflicts | unverified "
        "| true_med_px | true_p90_px"
    )

    for ratio in RADIUS_RATIOS:
        result = match_spots(
            source,
            target,
            source_eye.center,
            source_mouth,
            target_eye.center,
            target_mouth,
            radius_ratio=ratio,
        )
        proposals = set(result.pairs)
        recovered = proposals & references
        sources = {i for i, _ in references}
        targets = {j for _, j in references}
        conflicts = {
            (i, j)
            for i, j in proposals - references
            if i in sources or j in targets
        }
        unverified = len(proposals) - len(recovered) - len(conflicts)

        if references:
            indices = sorted(references)
            projected = project_points(source[[i for i, _ in indices]], result.matrix)
            expected = target[[j for _, j in indices]]
            errors = np.linalg.norm(projected - expected, axis=1)
            median = f"{np.median(errors):.1f}"
            p90 = f"{np.percentile(errors, 90):.1f}"
            verified_text = f"{len(recovered)}/{len(references)}"
            conflicts_text = str(len(conflicts))
        else:
            median = "n/a"
            p90 = "n/a"
            verified_text = "n/a"
            conflicts_text = "n/a"

        print(
            f"{ratio:.2f} | {ratio * eye_mouth_scale:.1f} | {len(proposals)} "
            f"| {verified_text} | {conflicts_text} | {unverified} "
            f"| {median} | {p90}"
        )


def run(project: Project) -> None:
    for subject_id, first_id, second_id in SAME_INDIVIDUAL_PAIRS:
        evaluate_pair(project, (subject_id, first_id), (subject_id, second_id))

    for source_label, target_label in DIFFERENT_INDIVIDUAL_PAIRS:
        evaluate_pair(project, source_label, target_label)


if __name__ == "__main__":
    run(Project.open())
