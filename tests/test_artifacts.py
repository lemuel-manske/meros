import numpy as np
from src.meros.adapters.fs_storage import MediaPaths
from src.meros.domain import TrackSelection
from src.meros.cmd import enhance_composites, build_composites


def test_selection_paths_include_interval_and_reference():
    paths = MediaPaths()
    a = TrackSelection('v', '0', 0, 10, 5)
    b = TrackSelection('v', '0', 0, 10, 6)
    c = TrackSelection('v', '0', 1, 10, 5)
    outputs = {paths.composite('v', '0', t.reference_frame, selection_id=t.selection_id) for t in [a, b, c]}
    assert len(outputs) == 3


def test_enhancement_completeness_does_not_require_preview(monkeypatch):
    from types import SimpleNamespace
    t = TrackSelection('v', '0', 0, 1, 0)
    monkeypatch.setattr(enhance_composites.metadata, 'read_individuals', lambda: [SimpleNamespace(tracks=[t])])
    monkeypatch.setattr(enhance_composites.media, 'composite_enhanced_exists', lambda *a, **k: True)
    monkeypatch.setattr(enhance_composites.media, 'composite_comparison_exists', lambda *a, **k: False)
    assert enhance_composites.enhanced_composites_complete()


def test_median_ignores_transparent_pixels():
    a = np.array([[[100, 100, 100, 255]]], dtype=np.uint8)
    b = np.zeros_like(a)
    assert np.array_equal(build_composites.median_composite([a, b]), a)
