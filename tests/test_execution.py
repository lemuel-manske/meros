from types import SimpleNamespace
from unittest.mock import Mock

from src.meros import metadata
from src.meros.cmd import extract_tracks, extract_frames
from src.meros.domain import Track


def test_current_video_manifest_loads():
    assert len(metadata.read_videos()) == 4
    assert all(v.frame_count is None for v in metadata.read_videos())


def test_tracking_uses_full_sequence_even_when_previews_exist(monkeypatch, tmp_path):
    for idx in range(3):
        (tmp_path / f'{idx}.jpg').write_bytes(b'frame')
    monkeypatch.setattr(extract_tracks, 'get_frame_ids', lambda _: [0, 1, 2])
    monkeypatch.setattr(extract_tracks.media, 'frame_path', lambda _, idx: tmp_path / f'{idx}.jpg')
    monkeypatch.setattr(extract_tracks.media, 'visualization_exists', lambda *args: True)
    seen = []
    def process(video, idx, obj_ids, logits, saved):
        seen.append(idx)
        saved.tracks['0'].frames[str(idx)] = object()
    monkeypatch.setattr(extract_tracks, 'process_frame', process)
    predictor = Mock()
    predictor.init_state.return_value = {}
    predictor.propagate_in_video.return_value = [(i, [], []) for i in range(3)]
    saved = extract_tracks.propagate_tracks('v', {'0': Track(0, [0, 0, 2, 2], {})}, predictor)
    assert seen == [0, 1, 2]
    assert len(saved.tracks['0'].frames) == 3


def test_partial_frame_cache_is_incomplete(monkeypatch):
    monkeypatch.setattr(extract_frames.metadata, 'read_videos', lambda: [SimpleNamespace(video_id='v', frame_count=3)])
    monkeypatch.setattr(extract_frames.media, 'read_frames', lambda _: [SimpleNamespace(frame_idx=0)])
    assert not extract_frames.frames_complete()
