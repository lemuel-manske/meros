import pytest

from meros.domain import Track
from meros.domain.prediction import Prediction
from meros.processing import extract_tracks, extract_frames
from meros.project import default_project
from tests.support import ProjectFixture, RecordedPredictor


class InterruptedPredictor(RecordedPredictor):
    def propagate_in_video(self, state: object):
        for prediction in super().propagate_in_video(state):
            yield prediction

            return


class DuplicateFramePredictor(RecordedPredictor):
    def propagate_in_video(self, state: object):
        for prediction in super().propagate_in_video(state):
            yield prediction

            yield prediction


class WrongSizePredictor(RecordedPredictor):
    def propagate_in_video(self, state: object):
        for prediction in super().propagate_in_video(state):
            yield Prediction(
                prediction.frame_idx, {key: mask[:1] for key, mask in prediction.masks.items()}
            )


@pytest.mark.parametrize(
    "predictor_type", [InterruptedPredictor, DuplicateFramePredictor, WrongSizePredictor]
)
def test_invalid_inference_does_not_replace_tracks(
    project_fixture: ProjectFixture, predictor_type: type[RecordedPredictor]
):
    project = project_fixture.project

    extract_frames.run(project)

    predictor = predictor_type(project_fixture.predictor.mask)

    with pytest.raises((RuntimeError, ValueError)):
        extract_tracks.run(project, predictor_factory=lambda: predictor)

    assert not project.metadata.paths.track("a").exists()


def test_tracking_uses_full_sequence_even_with_previews(project_fixture: ProjectFixture):
    project = project_fixture.project

    extract_frames.run(project)

    frame = project.media.read_frame("a", 0)

    for idx in range(3):
        project.media.write_visualization("a", idx, frame)

    saved = extract_tracks.propagate_tracks(
        "a", {"0": Track(0, [8, 8, 191, 151], {})}, project_fixture.predictor, project=project
    )

    assert saved.processed_frame_count == 3

    assert list(saved.tracks["0"].frames) == ["0", "1", "2"]

    assert project_fixture.predictor.states[0].frame_count == 3


def test_current_video_manifest_loads():
    assert len(default_project.metadata.read_videos()) == 4


def test_predictor_rejects_invalid_state(project_fixture: ProjectFixture):
    with pytest.raises(TypeError):
        list(project_fixture.predictor.propagate_in_video(object()))
