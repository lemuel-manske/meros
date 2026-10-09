from pathlib import Path

import pytest

from meros.adapters.fs_storage import write_json_atomic
from meros.pipeline.engine import Pipeline, Stage
from tests.support import ProjectFixture


class FileCalculation:
    """A real file-producing step with an observable completion contract."""

    def __init__(self, path: Path) -> None:
        self.path = path

        self.input = "input-1"

        self.calls = 0

    def run(self) -> None:
        self.calls += 1

        self.path.write_text(self.input)

    def complete(self) -> bool:
        return self.path.is_file() and self.path.read_text() == self.input

    def stage(self) -> Stage:
        return Stage(self.path.name, self.run, self.complete)


def test_completed_steps_are_reused(tmp_path: Path):
    calculation = FileCalculation(tmp_path / "output")

    pipeline = Pipeline([calculation.stage()])

    pipeline.run()

    pipeline.run()

    assert calculation.calls == 1

    calculation.input = "input-2"

    pipeline.run()

    assert calculation.path.read_text() == "input-2"

    assert calculation.calls == 2


def test_rebuilding_step_rebuilds_completed_downstream_outputs(tmp_path: Path):
    first = FileCalculation(tmp_path / "first")

    second = FileCalculation(tmp_path / "second")

    pipeline = Pipeline([first.stage(), second.stage()])

    pipeline.run()

    first.path.unlink()

    assert second.complete()

    pipeline.run()

    assert (first.calls, second.calls) == (2, 2)


def test_incomplete_step_stops_preparation(tmp_path: Path):
    downstream = FileCalculation(tmp_path / "downstream")

    pipeline = Pipeline([Stage("broken", lambda: None, lambda: False), downstream.stage()])

    with pytest.raises(RuntimeError, match="outputs are incomplete"):
        pipeline.run()

    assert downstream.calls == 0


def test_seed_change_rebuilds_tracking_and_downstream(prepared_project: ProjectFixture):
    project = prepared_project.project

    seeds = project.metadata.read_bboxes()

    seeds["a"]["0"][0] += 1

    write_json_atomic(project.metadata.paths.bboxes(), seeds)

    states = len(prepared_project.predictor.states)

    prepared_project.pipeline().run()

    saved = project.metadata.read_track("a")

    assert saved.tracks["0"].initial_bbox == seeds["a"]["0"]

    assert len(prepared_project.predictor.states) == states + 4

    assert not (project.media.paths.root / ".pipeline").exists()


def test_pipeline_repairs_composite_and_rebuilds_enhancement(prepared_project: ProjectFixture):
    project = prepared_project.project

    states = len(prepared_project.predictor.states)

    prepared_project.pipeline().run()

    assert len(prepared_project.predictor.states) == states

    selection = prepared_project.selections()[0]

    composite = project.media.paths.composite(selection)

    enhanced = project.media.paths.composite_enhanced(selection)

    expected = enhanced.read_bytes()

    image = project.media.read_composite_enhanced(selection)

    image[:, :, :3] = 0

    project.media.write_composite_enhanced(selection, image)

    composite.unlink()

    prepared_project.pipeline().run()

    assert enhanced.read_bytes() == expected

    assert len(prepared_project.predictor.states) == states
