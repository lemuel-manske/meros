import json

import numpy as np
import cv2 as cv
import pytest

from meros import metadata
from meros.experiments.migrate_002 import migrate


def legacy_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    identity = [
        {
            "individual_id": "a",
            "tracks": [
                {
                    "video_id": "v",
                    "track_id": "0",
                    "start_frame": 0,
                    "end_frame": 1,
                    "reference_frame": 0,
                }
            ],
        }
    ]

    manifest = tmp_path / "selections.json"

    manifest.write_text(json.dumps(identity))

    alignment = tmp_path / "data/metadata/alignment/v/0_0_1.json"

    alignment.parent.mkdir(parents=True)

    alignment.write_text(
        json.dumps(
            {
                "video_id": "v",
                "track_id": "0",
                "start_frame": 0,
                "end_frame": 1,
                "reference_frame": 0,
                "method": "baseline",
                "frames": [],
            }
        )
    )

    folder = tmp_path / "data/media/alignment/v/0"

    folder.mkdir(parents=True)

    image = np.full((8, 8, 4), 255, dtype=np.uint8)

    for name in ["0_median.png", "0_median_enhanced.png"]:
        cv.imwrite(str(folder / name), image)

    return manifest, folder


def test_migration_preserves_originals_and_is_repeatable(tmp_path, monkeypatch):
    manifest, folder = legacy_files(tmp_path, monkeypatch)

    monkeypatch.setattr(metadata, "individuals_path", None)

    receipt = tmp_path / "receipt.json"

    rows = migrate(manifest, receipt)

    assert len(rows) == 2

    assert len(list(folder.glob("*.png"))) == 2

    assert migrate(manifest, receipt) == rows

    assert json.loads(receipt.read_text())["originals_retained"]


def test_missing_legacy_file_does_not_partially_migrate(tmp_path, monkeypatch):
    manifest, folder = legacy_files(tmp_path, monkeypatch)

    monkeypatch.setattr(metadata, "individuals_path", None)

    (folder / "0_median_enhanced.png").unlink()

    with pytest.raises(FileNotFoundError):
        migrate(manifest, tmp_path / "receipt.json")

    assert not (tmp_path / "data/media/alignment/selections").exists()
