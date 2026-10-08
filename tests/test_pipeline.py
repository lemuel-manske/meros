from meros.pipeline import engine
from meros.pipeline.engine import Pipeline, Stage


def test_cache_reuses_computation_without_diagnostic_files(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "_STATE_FOLDER", tmp_path)
    calls = []
    stage = Stage("calculation", lambda: calls.append("run"), lambda: True, lambda: "input-1")
    pipeline = Pipeline([stage])
    pipeline.run("calculation")
    pipeline.run("calculation")
    assert calls == ["run"]
    pipeline.run("calculation", force=True)
    assert calls == ["run", "run"]


def test_seed_changes_invalidate_tracking(monkeypatch):
    seeds = {"v": {"0": [0, 0, 10, 10]}}
    monkeypatch.setattr(engine, "frames_stage_fingerprint", lambda: "frames")
    monkeypatch.setattr(engine, "file_fingerprint", lambda _: "checkpoint")
    monkeypatch.setattr(engine.metadata, "read_bboxes", lambda: seeds)
    before = engine.tracks_stage_fingerprint()
    seeds["v"]["0"][0] = 1
    assert before != engine.tracks_stage_fingerprint()
