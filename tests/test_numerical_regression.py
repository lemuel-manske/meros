"""Compare algorithms with the immutable pre-refactor implementation."""

import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import cv2 as cv
import numpy as np

from tests.support import Scenario, read_image

BASELINE = "386bdcec3b00ea348c523f1d3904c41210f175ba"


def load_baseline_module(name: str, path: str) -> types.ModuleType:
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], text=True)

    source = source.replace("from src.meros import media, metadata", "").replace(
        "from src.meros.domain import", "from _baseline_storage import"
    )

    module = types.ModuleType(name)

    sys.modules[name] = module

    exec(compile(source, f"<baseline:{path}>", "exec"), module.__dict__)

    return module


def legacy(name: str) -> types.ModuleType:
    if "_baseline_storage" not in sys.modules:
        load_baseline_module("_baseline_storage", "src/meros/domain/storage.py")

    return load_baseline_module(f"_baseline_{name}", f"src/meros/cmd/{name}.py")


def test_saved_representations_and_metrics_preserve_the_original_algorithms(
    scenario: Scenario, tmp_path: Path
) -> None:
    output = tmp_path / "run"

    rows = scenario.run(output)

    original_alignment = legacy("align_masked_crops")

    original_composite = legacy("build_composites")

    original_enhancement = legacy("enhance_composites")

    original_matching = legacy("match_composites")

    expected_images: dict[tuple[str, str], np.ndarray] = {}

    for selection in scenario.selections():
        reference = scenario.project.media.read_masked_crop(
            selection.video_id, selection.track_id, selection.reference_frame
        )

        crops = [reference]

        for idx in range(selection.start_frame, selection.end_frame + 1):
            if idx == selection.reference_frame:
                continue

            source = scenario.project.media.read_masked_crop(
                selection.video_id, selection.track_id, idx
            )

            matrix, _ = original_alignment.estimate_alignment(source, reference)

            if matrix is not None:
                crops.append(
                    cv.warpAffine(source, matrix, (reference.shape[1], reference.shape[0]))
                )

        composite = original_composite.median_composite(crops)

        enhanced = original_enhancement.enhance_contrast(composite)

        for representation, expected in {
            "reference": reference,
            "composite": composite,
            "enhanced_composite": enhanced,
        }.items():
            actual = read_image(
                output / "representations" / selection.selection_id / f"{representation}.png"
            )

            assert np.array_equal(actual, expected), (selection.selection_id, representation)

            expected_images[selection.selection_id, representation] = expected

    for row in rows:
        _, expected_stats = original_matching.match_composites(
            expected_images[row["source_selection_id"], row["representation"]],
            expected_images[row["target_selection_id"], row["representation"]],
        )

        assert {key: int(row[key]) for key in asdict(expected_stats)} == asdict(expected_stats)

        assert (row["ransac_attempted"] == "True") == (expected_stats.mutual_matches >= 8)
