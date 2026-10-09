"""Exercise evaluation and its report with real image matching and filesystem storage."""

import csv
import json
import re
from pathlib import Path

import cv2 as cv
import numpy as np

from meros.domain import Individual, TrackSelection
from meros.experiments.experiment_002 import evaluate
from meros.project import Project


def test_evaluation_reports_identical_and_featureless_pairs(tmp_path: Path) -> None:
    project = Project.open(tmp_path / "data", manifest=tmp_path / "selections.json")
    tracks = [TrackSelection("video", str(index), 0, 0, 0) for index in range(3)]
    individuals = [Individual("A", tracks[:2]), Individual("B", tracks[2:])]
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 256, (256, 256, 4), dtype=np.uint8)
    texture[:, :, 3] = 255
    blank = np.zeros_like(texture)
    blank[:, :, 3] = 255

    for track, image in zip(tracks, [texture, texture, blank], strict=True):
        project.media.write_masked_crop(track.video_id, track.track_id, 0, image)
        project.media.write_composite(track, image)
        project.media.write_composite_enhanced(track, image)

    output = tmp_path / "run"
    rows = evaluate(individuals, output, project=project)

    assert len(rows) == 9
    for row in rows:
        if row["same_individual"]:
            assert row["inliers"] > 0
            assert row["inliers"] == row["mutual_matches"]
            assert row["ransac_attempted"]
        else:
            assert row["inliers"] == row["mutual_matches"] == 0
            assert not row["ransac_attempted"]

    html = (output / "report.html").read_text()
    payload = re.search(r'<script id="results" type="application/json">(.*?)</script>', html)
    assert payload is not None
    report = json.loads(payload.group(1))
    assert report["rows"] == rows
    assert report["source_commit"] == json.loads((output / "run.json").read_text())["source_commit"]

    with (output / "metrics.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(rows)
    assert [int(row["inliers"]) for row in csv_rows] == [row["inliers"] for row in rows]

    for row in rows:
        for key in ("source_selection_id", "target_selection_id"):
            image_path = output / "representations" / row[key] / f"{row['representation']}.png"
            assert cv.imread(str(image_path), cv.IMREAD_UNCHANGED) is not None
