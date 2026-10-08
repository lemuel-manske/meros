from pathlib import Path

import pytest

from meros.adapters.fs_storage import write_json_atomic
from meros.pipeline.engine import Pipeline, Stage, tracks_stage_fingerprint
from tests.support import ProjectFixture


class FileCalculation:
    """A small file-producing stage for exercising engine orchestration."""

    def __init__(self, path: Path) -> None:
        self.path = path

        self.input = "input-1"

        self.calls = 0

    def run(self) -> None:
        self.calls += 1

        self.path.write_text(self.input)

    def complete(self) -> bool:
        return self.path.is_file()

    def fingerprint(self) -> str:
        return self.input

    def stage(self, name: str, dependencies: tuple[str, ...] = ()) -> Stage:
        return Stage(name, self.run, self.complete, self.fingerprint, dependencies)


def test_cache_reuses_and_rebuilds_outputs(tmp_path: Path):
    calculation = FileCalculation(tmp_path / "output")

    pipeline = Pipeline([calculation.stage("calculation")], state_folder=tmp_path / "state")

    pipeline.run("calculation")

    pipeline.run("calculation")

    assert calculation.calls == 1

    calculation.input = "input-2"

    pipeline.run("calculation")

    assert calculation.path.read_text() == "input-2"

    calculation.path.unlink()

    pipeline.run("calculation")

    pipeline.run("calculation", force=True)

    assert calculation.calls == 4


@pytest.mark.parametrize("payload", ["broken", "[]", '{"fingerprint": 4}'])
def test_corrupt_cache_state_is_recoverable(tmp_path: Path, payload: str):
    calculation = FileCalculation(tmp_path / "output")

    pipeline = Pipeline([calculation.stage("calculation")], state_folder=tmp_path)

    calculation.run()

    (tmp_path / "calculation.json").write_text(payload)

    pipeline.run("calculation")

    assert calculation.calls == 2


def test_seed_changes_invalidate_tracking(project_fixture: ProjectFixture):
    project = project_fixture.project

    before = tracks_stage_fingerprint(project)

    seeds = project.metadata.read_bboxes()

    seeds["a"]["0"][0] += 1

    write_json_atomic(project.metadata.paths.bboxes(), seeds)

    assert before != tracks_stage_fingerprint(project)


def test_real_pipeline_reuses_inference_and_repairs_deleted_composite(
    prepared_project: ProjectFixture,
):
    project = prepared_project.project

    states = len(prepared_project.predictor.states)

    prepared_project.pipeline().run("enhanced_composites")

    assert len(prepared_project.predictor.states) == states

    track = prepared_project.selections()[0]

    path = project.media.paths.composite(
        track.video_id, track.track_id, track.reference_frame, selection_id=track.selection_id
    )

    expected = path.read_bytes()

    path.unlink()

    prepared_project.pipeline().run("enhanced_composites")

    assert path.read_bytes() == expected

    assert len(prepared_project.predictor.states) == states


def test_engine_rejects_incomplete_stage_output(tmp_path: Path):
    stage = Stage("broken", lambda: None, lambda: False, lambda: "input")

    pipeline = Pipeline([stage], state_folder=tmp_path)

    with pytest.raises(RuntimeError, match="outputs are incomplete"):
        pipeline.run("broken")

    assert not (tmp_path / "broken.json").exists()


def test_engine_rejects_unknown_dependencies(tmp_path: Path):
    calculation = FileCalculation(tmp_path / "output")

    with pytest.raises(ValueError, match="unknown dependency"):
        Pipeline([calculation.stage("a", ("missing",))], state_folder=tmp_path)


def test_run_all_forces_shared_dependency_once(tmp_path: Path):
    source = FileCalculation(tmp_path / "source")

    first = FileCalculation(tmp_path / "first")

    second = FileCalculation(tmp_path / "second")

    pipeline = Pipeline(
        [
            first.stage("first", ("source",)),
            second.stage("second", ("source",)),
            source.stage("source"),
        ],
        state_folder=tmp_path / "state",
    )

    pipeline.run_all(force=True)

    assert (source.calls, first.calls, second.calls) == (1, 1, 1)


@pytest.mark.parametrize("dependencies", [("a",), ("b",)])
def test_engine_rejects_cycles(tmp_path: Path, dependencies: tuple[str, ...]):
    calculation = FileCalculation(tmp_path / "output")

    with pytest.raises(ValueError, match="Cyclic"):
        Pipeline(
            [
                calculation.stage("a", dependencies),
                calculation.stage("b", ("a",)),
            ],
            state_folder=tmp_path,
        )


def test_engine_rejects_duplicate_names(tmp_path: Path):
    calculation = FileCalculation(tmp_path / "output")

    with pytest.raises(ValueError, match="Duplicate"):
        Pipeline([calculation.stage("a"), calculation.stage("a")], state_folder=tmp_path)
