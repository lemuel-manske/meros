"""Prepare and compare the three representations used in experiment 002."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from collections.abc import Callable, Iterator, Sequence
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

import cv2 as cv

from meros.project import Project
from meros.domain import Individual, TrackSelection
from meros.domain.prediction import VideoPredictor
from meros.external import build_sam2_predictor
from meros.pipeline import create_pipeline
from meros.processing.match_images import match_images

type Representation = Literal["reference", "composite", "enhanced_composite"]


REPRESENTATIONS: tuple[Representation, ...] = ("reference", "composite", "enhanced_composite")


class ComparisonRow(TypedDict):
    source_individual_id: str
    target_individual_id: str

    source_selection_id: str
    target_selection_id: str

    same_individual: bool
    same_video: bool

    representation: Representation

    source_keypoints: int
    target_keypoints: int
    forward_good: int
    backward_good: int
    mutual_matches: int
    inliers: int
    ransac_attempted: bool


def iter_pairs(
    individuals: Sequence[Individual],
) -> Iterator[tuple[str, str, TrackSelection, TrackSelection]]:
    for individual in individuals:
        for a, b in combinations(individual.tracks, 2):
            yield individual.individual_id, individual.individual_id, a, b

    for source, target in combinations(individuals, 2):
        for a, b in product(source.tracks, target.tracks):
            yield source.individual_id, target.individual_id, a, b


def read_representations(
    track: TrackSelection, *, project: Project
) -> dict[Representation, NDArray[np.uint8]]:
    return {
        "reference": project.media.read_masked_crop(
            track.video_id, track.track_id, track.reference_frame
        ),
        "composite": project.media.read_composite(track),
        "enhanced_composite": project.media.read_composite_enhanced(track),
    }


def validate_individuals(individuals: Sequence[Individual]) -> None:
    if not individuals or any(not individual.tracks for individual in individuals):
        raise ValueError("Experiment requires nonempty individuals/selections")

    seen = set()

    identities = set()

    for individual in individuals:
        if individual.individual_id in identities:
            raise ValueError("Duplicate individual identity")

        identities.add(individual.individual_id)

        for track in individual.tracks:
            if track.selection_id in seen:
                raise ValueError("Selection assigned more than once")

            seen.add(track.selection_id)


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def evaluate(
    individuals: Sequence[Individual],
    output: Path,
    *,
    project: Project,
) -> list[ComparisonRow]:
    validate_individuals(individuals)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.parent / f".{output.name}-{uuid.uuid4().hex}.tmp"

    temporary.mkdir()

    try:
        images = {}

        for individual in individuals:
            for track in individual.tracks:
                for representation, image in read_representations(track, project=project).items():
                    path = (
                        temporary / "representations" / track.selection_id / f"{representation}.png"
                    )

                    path.parent.mkdir(parents=True, exist_ok=True)

                    if not cv.imwrite(str(path), image):
                        raise RuntimeError(f"Could not write {path}")

                    key = (track.selection_id, representation)

                    images[key] = image

        rows: list[ComparisonRow] = []

        for source_id, target_id, a, b in iter_pairs(individuals):
            for representation in REPRESENTATIONS:
                stats = match_images(
                    images[a.selection_id, representation],
                    images[b.selection_id, representation],
                )

                row: ComparisonRow = {
                    "source_individual_id": source_id,
                    "target_individual_id": target_id,
                    "source_selection_id": a.selection_id,
                    "target_selection_id": b.selection_id,
                    "same_individual": source_id == target_id,
                    "same_video": a.video_id == b.video_id,
                    "representation": representation,
                    "source_keypoints": stats.source_keypoints,
                    "target_keypoints": stats.target_keypoints,
                    "forward_good": stats.forward_good,
                    "backward_good": stats.backward_good,
                    "mutual_matches": stats.mutual_matches,
                    "inliers": stats.inliers,
                    "ransac_attempted": stats.ransac_attempted,
                }

                rows.append(row)

        if not rows:
            raise ValueError("No eligible comparison pairs")

        with (temporary / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))

            writer.writeheader()

            writer.writerows(rows)

        provenance = {"source_commit": git_revision()}

        (temporary / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")

        temporary.rename(output)

        return rows
    except BaseException:
        shutil.rmtree(temporary)

        raise


def run(
    project: Project,
    output: Path,
    *,
    predictor_factory: Callable[[Path], VideoPredictor] = build_sam2_predictor,
) -> list[ComparisonRow]:
    individuals = project.metadata.read_individuals()

    validate_individuals(individuals)

    create_pipeline(project, predictor_factory=predictor_factory).run()

    rows = evaluate(individuals, output, project=project)

    print(f"Wrote {len(rows)} comparisons to {output}")

    return rows


def main() -> None:
    project = Project.open(manifest=Path("experiments/002/selections.json"))

    output = Path("results/002") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )

    run(project, output)


if __name__ == "__main__":
    main()
