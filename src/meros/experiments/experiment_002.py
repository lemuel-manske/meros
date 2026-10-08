"""Compare three representations without requiring diagnostic image caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path

import cv2 as cv
import numpy as np

from meros import media, metadata
from meros.config import options
from meros.processing.match_images import match_images

REPRESENTATIONS = ("reference", "composite", "enhanced_composite")


def iter_pairs(individuals, *, cross_video_only=False):
    for individual in individuals:
        for a, b in combinations(individual.tracks, 2):
            if not cross_video_only or a.video_id != b.video_id:
                yield individual.individual_id, individual.individual_id, a, b

    for source, target in combinations(individuals, 2):
        for a, b in product(source.tracks, target.tracks):
            if not cross_video_only or a.video_id != b.video_id:
                yield source.individual_id, target.individual_id, a, b


def read_representation(track, representation):
    args = (track.video_id, track.track_id, track.reference_frame)

    if representation == "reference":
        return media.read_masked_crop(*args)

    if representation == "composite":
        return media.read_composite(*args, selection_id=track.selection_id)

    return media.read_composite_enhanced(*args, selection_id=track.selection_id)


def validate_individuals(individuals):
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


def git_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)

    return result.stdout.strip() if result.returncode == 0 else None


def evaluate(individuals, output: Path, *, cross_video_only=False, preparation_fingerprint=None):
    validate_individuals(individuals)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.parent / f".{output.name}-{uuid.uuid4().hex}.tmp"

    temporary.mkdir()

    try:
        images = {}

        image_hashes = {}

        for individual in individuals:
            for track in individual.tracks:
                for representation in REPRESENTATIONS:
                    image = read_representation(track, representation)

                    path = (
                        temporary / "representations" / track.selection_id / f"{representation}.png"
                    )

                    path.parent.mkdir(parents=True, exist_ok=True)

                    if not cv.imwrite(str(path), image):
                        raise RuntimeError(f"Could not write {path}")

                    key = (track.selection_id, representation)

                    images[key] = image

                    image_hashes[str(path.relative_to(temporary))] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()

        rows = []

        for source_id, target_id, a, b in iter_pairs(
            individuals, cross_video_only=cross_video_only
        ):
            for representation in REPRESENTATIONS:
                canvas, stats = match_images(
                    images[a.selection_id, representation],
                    images[b.selection_id, representation],
                    draw=options.diagnostics,
                )

                row = {
                    "source_individual_id": source_id,
                    "target_individual_id": target_id,
                    "source_selection_id": a.selection_id,
                    "target_selection_id": b.selection_id,
                    "same_individual": source_id == target_id,
                    "same_video": a.video_id == b.video_id,
                    "representation": representation,
                    **asdict(stats),
                }

                rows.append(row)

                if canvas is not None:
                    path = (
                        temporary
                        / "diagnostics"
                        / f"{a.selection_id}__{b.selection_id}"
                        / f"{representation}.png"
                    )

                    path.parent.mkdir(parents=True, exist_ok=True)

                    if not cv.imwrite(str(path), canvas):
                        raise RuntimeError(f"Could not write {path}")

        if not rows:
            raise ValueError("No eligible comparison pairs")

        with (temporary / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))

            writer.writeheader()

            writer.writerows(rows)

        from meros.processing import (
            match_images as matching,
            align_masked_crops,
            enhance_composites,
        )

        def constants(module):
            return {
                name: value
                for name, value in vars(module).items()
                if name.isupper() and isinstance(value, (int, float, str, bool))
            }

        provenance = {
            "schema_version": 1,
            "experiment": "002",
            "source_commit": git_revision(),
            "source_sha256": {
                str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))
            },
            "python": sys.version,
            "opencv": cv.__version__,
            "numpy": np.__version__,
            "selections": [asdict(ind) for ind in individuals],
            "cross_video_only": cross_video_only,
            "rows": len(rows),
            "image_sha256": image_hashes,
            "preparation_fingerprint": preparation_fingerprint,
            "preparation_parameters_verified": preparation_fingerprint is not None,
            "configuration": {
                "matching": constants(matching),
                "alignment": constants(align_masked_crops),
                "enhancement": constants(enhance_composites),
            },
        }

        (temporary / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")

        temporary.rename(output)

        return rows
    except BaseException:
        shutil.rmtree(temporary)

        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Use already prepared selection-scoped representations; no SAM2 imports",
    )

    parser.add_argument(
        "--diagnostics", action="store_true", help="Write optional overlays/previews/match drawings"
    )

    parser.add_argument(
        "--force", action="store_true", help="Recompute preparation stages, including diagnostics"
    )

    parser.add_argument("--cross-video-only", action="store_true")

    parser.add_argument("--manifest", type=Path, default=Path("experiments/002/selections.json"))

    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    options.diagnostics = args.diagnostics

    metadata.individuals_path = args.manifest

    individuals = metadata.read_individuals()

    validate_individuals(individuals)

    preparation_fingerprint = None

    if not args.evaluate_only:
        from meros.pipeline import pipeline

        pipeline.run("enhanced_composites", force=args.force)

        preparation_fingerprint = pipeline.stages["enhanced_composites"].fingerprint()

    output = args.output or Path("results/002") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )

    rows = evaluate(
        individuals,
        output,
        cross_video_only=args.cross_video_only,
        preparation_fingerprint=preparation_fingerprint,
    )

    print(f"Wrote {len(rows)} comparisons to {output}")


if __name__ == "__main__":
    main()
