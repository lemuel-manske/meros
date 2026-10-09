import numpy as np

from meros.adapters.fs_storage import MediaPaths
from meros.domain import TrackSelection
from meros.processing import build_composites
from tests.support import ProjectFixture


def test_selection_paths_include_interval_and_reference():
    paths = MediaPaths()

    selections = [
        TrackSelection("v", "0", 0, 10, 5),
        TrackSelection("v", "0", 0, 10, 6),
        TrackSelection("v", "0", 1, 10, 5),
    ]

    outputs = {paths.composite(track) for track in selections}

    assert len(outputs) == 3


def test_median_ignores_transparent_pixels():
    visible = np.array([[[100, 100, 100, 255]]], dtype=np.uint8)

    transparent = np.zeros_like(visible)

    assert np.array_equal(build_composites.median_composite([visible, transparent]), visible)


def test_preparation_keeps_transforms_without_intermediate_images(prepared_project: ProjectFixture):
    project = prepared_project.project

    for track in prepared_project.selections():
        alignment = project.metadata.read_alignment(track.id)

        assert all(row.status == "accepted" for row in alignment.frames)

        directory = project.media.paths.composite(track).parent

        assert {path.name for path in directory.iterdir()} == {
            "composite.png",
            "enhanced_composite.png",
        }

    assert not (project.media.paths.root / "media/visualizations").exists()
