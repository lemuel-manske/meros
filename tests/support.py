"""Real filesystem fixtures with an explicit CPU substitute for GPU inference."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

from meros.adapters.fs_storage import write_json_atomic
from meros.domain import Individual, TrackSelection, Video
from meros.domain.prediction import BinaryMask, Prediction, SeedBox
from meros.pipeline.engine import Pipeline, create_pipeline
from meros.project import Project


@dataclass
class RecordedState:
    frame_count: int

    object_ids: list[str] = field(default_factory=list)


class RecordedPredictor:
    """Replay a known segmentation while using the real frames and seed API.

    This replaces only SAM2/CUDA. Tracking, crops, alignment, composites,
    matching, serialization, and cache decisions run their production code.
    """

    def __init__(self, mask: BinaryMask) -> None:
        self.mask = mask

        self.states: list[RecordedState] = []

    def init_state(self, video_path: str) -> RecordedState:
        paths = list(Path(video_path).glob("*.jpg"))

        if not paths or any(cv.imread(str(path)) is None for path in paths):
            raise ValueError("Recorded inference requires readable frames")

        state = RecordedState(len(paths))

        self.states.append(state)

        return state

    def add_new_points_or_box(
        self, inference_state: object, frame_idx: int, obj_id: int, box: SeedBox
    ) -> None:
        if not isinstance(inference_state, RecordedState):
            raise TypeError("Expected a recorded inference state")

        if not 0 <= frame_idx < inference_state.frame_count or box.shape != (4,):
            raise ValueError("Invalid seed")

        inference_state.object_ids.append(str(obj_id))

    def propagate_in_video(self, state: object) -> Iterator[Prediction]:
        if not isinstance(state, RecordedState):
            raise TypeError("Expected a recorded inference state")

        for idx in range(state.frame_count):
            yield Prediction(idx, {obj_id: self.mask.copy() for obj_id in state.object_ids})


@dataclass
class ProjectFixture:
    project: Project

    predictor: RecordedPredictor

    individuals: list[Individual]

    def pipeline(self) -> Pipeline:
        return create_pipeline(self.project, predictor_factory=lambda: self.predictor)

    def prepare(self) -> None:
        self.pipeline().run("enhanced_composites")

    def selections(self) -> list[TrackSelection]:
        return [track for individual in self.individuals for track in individual.tracks]

    def empty_selections(self) -> None:
        assert self.project.metadata.individuals_path is not None

        write_json_atomic(self.project.metadata.individuals_path, [])


def create_project(root: Path) -> ProjectFixture:
    manifest = root / "selections.json"

    checkpoint = root / "recorded-segmentation"

    root.mkdir(parents=True, exist_ok=True)

    checkpoint.write_text("Deterministic CPU segmentation fixture\n")

    project = Project.open(root, manifest=manifest, checkpoint=checkpoint)

    random = np.random.default_rng(7)

    image = random.integers(0, 256, (160, 200, 3), dtype=np.uint8)

    mask = np.zeros((160, 200), dtype=np.bool_)

    mask[8:152, 8:192] = True

    videos = []

    for video_id in ("a", "b"):
        path = root / f"{video_id}.avi"

        writer = cv.VideoWriter(str(path), cv.VideoWriter.fourcc(*"MJPG"), 5, (200, 160))

        if not writer.isOpened():
            raise RuntimeError("Could not create fixture video")

        try:
            for _ in range(3):
                writer.write(image)
        finally:
            writer.release()

        videos.append(Video(video_id, False, path.name, str(path), 3))

    individuals = [
        Individual("first", [TrackSelection("a", "0", 0, 2, 0), TrackSelection("b", "0", 0, 2, 0)]),
        Individual("second", [TrackSelection("a", "1", 0, 2, 1)]),
    ]

    write_json_atomic(project.metadata.paths.videos(), [asdict(video) for video in videos])

    write_json_atomic(
        project.metadata.paths.bboxes(),
        {
            "a": {"0": [8, 8, 191, 151], "1": [8, 8, 191, 151]},
            "b": {"0": [8, 8, 191, 151]},
        },
    )

    write_json_atomic(manifest, [asdict(individual) for individual in individuals])

    return ProjectFixture(project, RecordedPredictor(mask), individuals)
