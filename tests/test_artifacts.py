import numpy as np
from meros.adapters.fs_storage import MediaPaths
from meros.domain import TrackSelection
from meros.processing import enhance_composites, build_composites


def test_selection_paths_include_interval_and_reference():
    paths = MediaPaths()
    a = TrackSelection("v", "0", 0, 10, 5)
    b = TrackSelection("v", "0", 0, 10, 6)
    c = TrackSelection("v", "0", 1, 10, 5)
    outputs = {
        paths.composite("v", "0", t.reference_frame, selection_id=t.selection_id) for t in [a, b, c]
    }
    assert len(outputs) == 3


def test_enhancement_completeness_does_not_require_preview(monkeypatch):
    from types import SimpleNamespace

    t = TrackSelection("v", "0", 0, 1, 0)
    monkeypatch.setattr(
        enhance_composites.metadata, "read_individuals", lambda: [SimpleNamespace(tracks=[t])]
    )
    monkeypatch.setattr(enhance_composites.media, "composite_enhanced_exists", lambda *a, **k: True)
    monkeypatch.setattr(
        enhance_composites.media, "composite_comparison_exists", lambda *a, **k: False
    )
    assert enhance_composites.enhanced_composites_complete()


def test_median_ignores_transparent_pixels():
    a = np.array([[[100, 100, 100, 255]]], dtype=np.uint8)
    b = np.zeros_like(a)
    assert np.array_equal(build_composites.median_composite([a, b]), a)


def test_alignment_and_composite_need_no_saved_aligned_images(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from meros.processing import align_masked_crops
    from meros.domain import AlignmentMetadata
    from meros.config import options

    image = np.full((32, 32, 4), 255, dtype=np.uint8)
    selection = TrackSelection("v", "0", 0, 1, 0)
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rows = []
    monkeypatch.setattr(options, "diagnostics", False)
    monkeypatch.setattr(align_masked_crops.media, "read_masked_crop", lambda *args: image)
    monkeypatch.setattr(align_masked_crops, "extract_features", lambda *args: None)
    monkeypatch.setattr(
        align_masked_crops,
        "estimate_alignment_from_features",
        lambda *args: (
            matrix,
            AlignmentMetadata("accepted", 12, 12, source_to_reference=matrix.tolist()),
        ),
    )
    monkeypatch.setattr(
        align_masked_crops.metadata,
        "write_alignment",
        lambda key, reference, results: rows.extend(results),
    )
    fail_write = Mock(side_effect=AssertionError("Unexpected intermediate image write"))
    monkeypatch.setattr(align_masked_crops.media, "write_aligned_crop", fail_write)
    monkeypatch.setattr(align_masked_crops.media, "write_aligned_overlay", fail_write)
    assert align_masked_crops.align_sequence(selection.id, selection.reference_frame) == (1, 1)
    monkeypatch.setattr(
        build_composites.metadata,
        "read_alignment",
        lambda _: SimpleNamespace(start_frame=0, end_frame=1, reference_frame=0, frames=rows),
    )
    captured = []
    monkeypatch.setattr(
        build_composites.media, "write_composite", lambda *a, **kw: captured.append(a[3])
    )
    assert build_composites.build_composite(selection) == 2
    assert np.array_equal(captured[0], image)
    fail_write.assert_not_called()
