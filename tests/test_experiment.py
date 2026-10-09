"""Observable experiment behavior through videos, inference, processing, and saved runs."""

import json
import os
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from meros.adapters.fs_storage import write_json_atomic
from meros.domain import TrackSelection
from meros.experiments import experiment_002 as experiment
from meros.external.sam2 import SAM2Predictor
from tests.support import InterruptedSam2, Scenario, create_scenario, read_image


OLD_TIME = 1_700_000_000_000_000_000


def mark_artifacts(scenario: Scenario) -> dict[str, list[Path]]:
    artifacts = scenario.artifacts()

    for paths in artifacts.values():
        for path in paths:
            os.utime(path, ns=(OLD_TIME, OLD_TIME))

    return artifacts


def snapshot(output: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }


def test_run_saves_all_pairs_and_the_actual_representations(scenario: Scenario, tmp_path: Path):
    output = tmp_path / "run"

    rows = scenario.run(output)

    selections = scenario.selections()

    expected_pairs = {
        tuple(selection.selection_id for selection in pair) for pair in combinations(selections, 2)
    }

    assert {
        (row["source_selection_id"], row["target_selection_id"], row["representation"])
        for row in rows
    } == {
        (*pair, representation)
        for pair in expected_pairs
        for representation in ("reference", "composite", "enhanced_composite")
    }

    assert len(rows) == 18

    assert sum(row["same_individual"] == "True" for row in rows) == 9

    assert sum(row["same_video"] == "True" for row in rows) == 9

    assert all(int(row["inliers"]) > 0 for row in rows if row["same_individual"] == "True")

    identities = {
        selection.selection_id: individual.individual_id
        for individual in scenario.project.metadata.read_individuals()
        for selection in individual.tracks
    }

    videos = {selection.selection_id: selection.video_id for selection in selections}

    for row in rows:
        source, target = row["source_selection_id"], row["target_selection_id"]

        assert (row["same_individual"] == "True") == (identities[source] == identities[target])

        assert (row["same_video"] == "True") == (videos[source] == videos[target])

    for selection in selections:
        saved = output / "representations" / selection.selection_id

        assert {path.name for path in saved.iterdir()} == {
            "reference.png",
            "composite.png",
            "enhanced_composite.png",
        }

        assert np.array_equal(
            read_image(saved / "reference.png"),
            scenario.project.media.read_masked_crop(
                selection.video_id, selection.track_id, selection.reference_frame
            ),
        )

        assert np.array_equal(
            read_image(saved / "composite.png"), scenario.project.media.read_composite(selection)
        )

        assert np.array_equal(
            read_image(saved / "enhanced_composite.png"),
            scenario.project.media.read_composite_enhanced(selection),
        )

    crop = read_image(output / "representations" / selections[0].selection_id / "reference.png")

    frame = scenario.project.media.read_frame("a", 0)

    assert crop.shape == (160, 200, 4)

    assert np.array_equal(crop[-1, -1, :3], frame[167, 207])

    assert crop[-1, -1, 3] == 255

    assert np.all(crop[40:56, 40:56] == 0)

    composite = read_image(
        output / "representations" / selections[0].selection_id / "composite.png"
    )

    colors = [scenario.project.media.read_frame("a", idx)[56, 56].astype(float) for idx in (1, 2)]

    assert np.array_equal(composite[48, 48, :3], np.rint(np.mean(colors, axis=0)).astype(np.uint8))

    assert composite[48, 48, 3] == 255

    assert json.loads((output / "run.json").read_text()) == {
        "source_commit": experiment.git_revision()
    }


def test_second_run_reuses_preparation_and_preserves_the_first_run(
    scenario: Scenario, tmp_path: Path
):
    first = tmp_path / "first"

    rows = scenario.run(first)

    saved = snapshot(first)

    artifacts = mark_artifacts(scenario)

    assert scenario.run(tmp_path / "second") == rows

    assert snapshot(first) == saved

    assert all(
        path.stat().st_mtime_ns == OLD_TIME for paths in artifacts.values() for path in paths
    )

    with pytest.raises(FileExistsError):
        scenario.run(first)

    assert snapshot(first) == saved


@pytest.mark.parametrize(
    "stage",
    [
        pytest.param("frames", id="interrupted-frame-write"),
        pytest.param("tracks", id="unfinished-tracking"),
        pytest.param("masked_crops", id="empty-segmentation"),
        pytest.param("alignments", id="invalid-transform"),
        pytest.param("composites", id="missing-composite"),
        pytest.param("enhanced_composites", id="truncated-enhancement"),
    ],
)
def test_damaged_preparation_is_rebuilt_through_the_final_result(
    scenario: Scenario, tmp_path: Path, stage: str
):
    first = tmp_path / "first"

    expected_rows = scenario.run(first)

    expected_images = {
        path: data for path, data in snapshot(first).items() if path.suffix == ".png"
    }

    artifacts = mark_artifacts(scenario)

    path = artifacts[stage][0]

    if stage == "composites":
        path.unlink()
    elif stage == "tracks":
        payload = json.loads(path.read_text())

        payload["processed_frame_count"] = 1

        write_json_atomic(path, payload)
    elif stage == "alignments":
        payload = json.loads(path.read_text())

        payload["frames"][0]["source_to_reference"] = [[0, 0, 0], [0, 0, 0]]

        write_json_atomic(path, payload)
    elif stage == "masked_crops":
        image = read_image(path)

        image[:, :, 3] = 0

        assert cv.imwrite(str(path), image)
    elif stage == "enhanced_composites":
        assert cv.imwrite(str(path), read_image(path)[:1])
    else:
        path.write_bytes(b"interrupted JPEG write")

    # A stale downstream image remains structurally valid. The rebuilt run
    # must replace it because an upstream result changed.
    if stage != "enhanced_composites":
        downstream = artifacts["enhanced_composites"][-1]

        image = read_image(downstream)

        image[:, :, :3] = 0

        assert cv.imwrite(str(downstream), image)

        os.utime(downstream, ns=(OLD_TIME, OLD_TIME))

    rebuilt = tmp_path / "rebuilt"

    assert scenario.run(rebuilt) == expected_rows

    assert {
        path: data for path, data in snapshot(rebuilt).items() if path.suffix == ".png"
    } == expected_images

    stages = list(artifacts)

    for name, paths in artifacts.items():
        if stages.index(name) < stages.index(stage):
            assert all(path.stat().st_mtime_ns == OLD_TIME for path in paths)
        else:
            assert all(path.stat().st_mtime_ns != OLD_TIME for path in paths)


def test_seed_change_updates_the_saved_representations_without_reextracting_video(
    scenario: Scenario, tmp_path: Path
):
    first = tmp_path / "first"

    scenario.run(first)

    saved = snapshot(first)

    artifacts = mark_artifacts(scenario)

    boxes = scenario.project.metadata.read_bboxes()

    boxes["a"]["0"][0] = 24

    write_json_atomic(scenario.project.metadata.paths.bboxes(), boxes)

    second = tmp_path / "second"

    scenario.run(second)

    selection = scenario.selections()[0]

    reference = read_image(second / "representations" / selection.selection_id / "reference.png")

    assert reference.shape == (160, 184, 4)

    assert np.array_equal(
        reference[-1, -1, :3], scenario.project.media.read_frame("a", 0)[167, 207]
    )

    assert (
        read_image(second / "representations" / selection.selection_id / "composite.png").shape
        == reference.shape
    )

    assert all(path.stat().st_mtime_ns == OLD_TIME for path in artifacts["frames"])

    assert all(
        path.stat().st_mtime_ns != OLD_TIME
        for name, paths in artifacts.items()
        if name != "frames"
        for path in paths
    )

    assert snapshot(first) == saved


def test_selection_change_uses_the_new_interval_without_overwriting_other_selections(
    scenario: Scenario, tmp_path: Path
):
    scenario.run(tmp_path / "first")

    artifacts = mark_artifacts(scenario)

    individuals = scenario.project.metadata.read_individuals()

    previous = individuals[0].tracks[0]

    new = TrackSelection("a", "0", 0, 0, 0)

    individuals[0].tracks[0] = new

    other = scenario.project.media.paths.composite(individuals[0].tracks[1]).read_bytes()

    write_json_atomic(
        scenario.project.metadata.individuals_path,
        [asdict(individual) for individual in individuals],
    )

    output = tmp_path / "second"

    rows = scenario.run(output)

    ids = {row[key] for row in rows for key in ("source_selection_id", "target_selection_id")}

    assert new.selection_id in ids

    assert previous.selection_id not in ids

    assert np.array_equal(
        read_image(output / "representations" / new.selection_id / "composite.png"),
        read_image(output / "representations" / new.selection_id / "reference.png"),
    )

    assert scenario.project.media.paths.composite(individuals[0].tracks[1]).read_bytes() == other

    assert all(
        path.stat().st_mtime_ns == OLD_TIME
        for name in ("frames", "tracks", "masked_crops")
        for path in artifacts[name]
    )


def test_interrupted_tracking_preserves_saved_work_and_can_be_retried(
    scenario: Scenario, tmp_path: Path
):
    first = tmp_path / "first"

    scenario.run(first)

    saved = snapshot(first)

    artifacts = {
        path: path.read_bytes() for paths in scenario.artifacts().values() for path in paths
    }

    boxes = scenario.project.metadata.read_bboxes()

    boxes["a"]["0"][0] = 24

    write_json_atomic(scenario.project.metadata.paths.bboxes(), boxes)

    output = tmp_path / "retry"

    with pytest.raises(RuntimeError, match="full sequence"):
        experiment.run(
            scenario.project,
            output,
            predictor_factory=lambda checkpoint: SAM2Predictor(InterruptedSam2()),
        )

    assert not output.exists()

    assert {path: path.read_bytes() for path in artifacts} == artifacts

    assert snapshot(first) == saved

    assert len(scenario.run(output)) == 18

    assert (
        scenario.project.metadata.paths.track("a").read_bytes()
        != artifacts[scenario.project.metadata.paths.track("a")]
    )


def test_failed_evaluation_publishes_nothing_and_preparation_repairs_it(
    scenario: Scenario, tmp_path: Path
):
    scenario.run(tmp_path / "first")

    selection = scenario.selections()[0]

    scenario.project.media.paths.composite(selection).unlink()

    output = tmp_path / "retry"

    with pytest.raises(FileNotFoundError):
        experiment.evaluate(
            scenario.project.metadata.read_individuals(), output, project=scenario.project
        )

    assert not output.exists()

    assert not list(tmp_path.glob(".retry-*.tmp"))

    assert len(scenario.run(output)) == 18


def test_frames_without_features_do_not_darken_the_composite(tmp_path: Path):
    scenario = create_scenario(tmp_path / "data", blank_last_frame=True)

    output = tmp_path / "run"

    scenario.run(output)

    selection = scenario.selections()[0]

    composite = read_image(output / "representations" / selection.selection_id / "composite.png")

    assert np.array_equal(composite[48, 48, :3], scenario.project.media.read_frame("a", 1)[56, 56])

    assert composite[48, 48, 3] == 255
