import csv
import json
from pathlib import Path

import numpy as np
import pytest

from meros.experiments import experiment_002 as exp
from meros.experiments.__main__ import main as run_experiment
from meros.processing.match_images import match_images
from meros.project import Project
from tests.support import ProjectFixture


def test_current_manifest_has_21_pairs_without_filename_collisions():
    individuals = Project.open(
        manifest=Path("experiments/002/selections.json")
    ).metadata.read_individuals()

    pairs = list(exp.iter_pairs(individuals))

    assert len(pairs) == 21

    assert sum(a == b for a, b, _, _ in pairs) == 3

    assert len({(a.selection_id, b.selection_id) for _, _, a, b in pairs}) == 21

    assert len(list(exp.iter_pairs(individuals, cross_video_only=True))) == 18


def test_pipeline_evaluation_persists_real_representations(
    prepared_project: ProjectFixture, tmp_path: Path
):
    project = prepared_project.project

    output = tmp_path / "run"

    rows = exp.evaluate(prepared_project.individuals, output, project=project)

    assert len(rows) == 9

    assert len(list((output / "representations").rglob("*.png"))) == 9

    assert not (output / "diagnostics").exists()

    with (output / "metrics.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 9

    assert set(json.loads((output / "run.json").read_text())) == {"source_commit"}

    assert all(row["inliers"] > 0 for row in rows)

    with pytest.raises(FileExistsError):
        exp.evaluate(prepared_project.individuals, output, project=project)


def test_failed_evaluation_leaves_no_partial_run(prepared_project: ProjectFixture, tmp_path: Path):
    project = prepared_project.project

    selection = prepared_project.selections()[0]

    project.media.paths.composite(
        *selection.id[:2], selection.reference_frame, selection_id=selection.selection_id
    ).unlink()

    with pytest.raises(FileNotFoundError):
        exp.evaluate(prepared_project.individuals, tmp_path / "run", project=project)

    assert not (tmp_path / "run").exists()

    assert not list(tmp_path.glob(".run-*.tmp"))


@pytest.mark.parametrize("identifier", ["does_not_exist", "../../002", "002.bad"])
def test_runner_rejects_invalid_experiment(identifier: str):
    with pytest.raises(SystemExit) as error:
        run_experiment([identifier])

    assert error.value.code == 2


def test_matcher_skips_ransac_on_blank_image():
    image = np.zeros((32, 32, 4), dtype=np.uint8)

    image[:, :, 3] = 255

    canvas, stats = match_images(image, image)

    assert canvas is None

    assert stats.mutual_matches == 0

    assert stats.inliers == 0

    assert not stats.ransac_attempted


def test_runner_forwards_experiment_help(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as error:
        run_experiment(["002", "--help"])

    assert error.value.code == 0

    assert "--evaluate-only" in capsys.readouterr().out


def test_evaluate_only_cli_reads_real_files(prepared_project: ProjectFixture, tmp_path: Path):
    manifest = prepared_project.project.metadata.individuals_path

    assert manifest is not None

    output = tmp_path / "cli-run"

    exp.main(
        [
            "--evaluate-only",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        project=prepared_project.project,
    )

    assert (output / "metrics.csv").is_file()

    assert (output / "run.json").is_file()


def test_diagnostics_are_optional_after_real_generation(
    project_fixture: ProjectFixture, tmp_path: Path
):
    project = project_fixture.project

    project.options.diagnostics = True

    project_fixture.prepare()

    output = tmp_path / "diagnostic-run"

    exp.evaluate(project_fixture.individuals, output, project=project)

    assert len(list((output / "diagnostics").rglob("*.png"))) == 9

    assert len(list((project.media.paths.root / "media/visualizations").rglob("*.jpg"))) == 6

    for track in project_fixture.selections():
        path = project.media.paths.composite(
            track.video_id, track.track_id, track.reference_frame, selection_id=track.selection_id
        )

        for diagnostic in path.parent.glob("*.png"):
            if diagnostic.name not in {"composite.png", "enhanced_composite.png"}:
                diagnostic.unlink()

    states = len(project_fixture.predictor.states)

    project.options.diagnostics = False

    project_fixture.pipeline().run("enhanced_composites")

    assert len(project_fixture.predictor.states) == states
