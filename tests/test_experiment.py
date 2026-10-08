import csv
import json
import numpy as np
import pytest
from meros import metadata
from meros.experiments import experiment_002 as exp
from meros.processing.match_images import match_images


def test_current_manifest_has_21_pairs_without_filename_collisions():
    individuals = metadata.read_individuals()
    pairs = list(exp.iter_pairs(individuals))
    assert len(pairs) == 21
    assert sum(a == b for a, b, _, _ in pairs) == 3
    assert len({(a.selection_id, b.selection_id) for _, _, a, b in pairs}) == 21
    assert len(list(exp.iter_pairs(individuals, cross_video_only=True))) == 18


def test_evaluation_persists_all_representations_and_metrics(monkeypatch, tmp_path):
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    monkeypatch.setattr(exp, "read_representation", lambda *args: image)
    output = tmp_path / "run"
    rows = exp.evaluate(metadata.read_individuals(), output)
    assert len(rows) == 63
    assert len(list((output / "representations").rglob("*.png"))) == 21
    assert not (output / "diagnostics").exists()
    assert len(list(csv.DictReader((output / "metrics.csv").open()))) == 63
    assert len(json.loads((output / "run.json").read_text())["image_sha256"]) == 21
    with pytest.raises(FileExistsError):
        exp.evaluate(metadata.read_individuals(), output)


def test_matcher_does_not_draw_or_claim_ransac_on_blank_image():
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    canvas, stats = match_images(image, image)
    assert canvas is None
    assert stats.mutual_matches == 0
    assert stats.inliers == 0
    assert not stats.ransac_attempted
