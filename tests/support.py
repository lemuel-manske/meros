"""A real video experiment with a fake only at the SAM2/CUDA boundary."""

import csv
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from meros.adapters.fs_storage import write_json_atomic
from meros.domain import Individual, TrackSelection, Video
from meros.domain.prediction import BinaryMask, SeedBox
from meros.experiments import experiment_002
from meros.external.sam2 import MaskLogits, SAM2Predictor
from meros.project import Project


type Image = NDArray[np.uint8]


@dataclass
class MaskTensor:
    data: BinaryMask

    def cpu(self) -> "MaskTensor":
        return self

    def numpy(self) -> BinaryMask:
        return self.data


@dataclass
class LogitsTensor:
    data: NDArray[np.float32]

    def gt(self, threshold: float) -> MaskTensor:
        return MaskTensor(self.data > threshold)


@dataclass
class VideoState:
    frames: list[Image]

    boxes: dict[int, SeedBox] = field(default_factory=dict)


class RecordedSam2:
    """Emit logits for the seeded regions; production code handles everything else.

    A hole in the first frame makes aggregation distinguish transparent pixels
    from visible black pixels. Zero logits inside that hole exercise the actual
    SAM2 adapter's threshold and singleton-channel conversion.
    """

    def init_state(
        self, video_path: str, *, offload_video_to_cpu: bool, offload_state_to_cpu: bool
    ) -> VideoState:
        paths = sorted(Path(video_path).glob("*.jpg"), key=lambda path: int(path.stem))

        return VideoState([read_image(path) for path in paths])

    def add_new_points_or_box(
        self, *, inference_state: object, frame_idx: int, obj_id: int, box: SeedBox
    ) -> object:
        assert isinstance(inference_state, VideoState)

        inference_state.boxes[obj_id] = box

        return None

    def propagate_in_video(
        self, state: object
    ) -> Iterator[tuple[int, list[int], list[MaskLogits]]]:
        assert isinstance(state, VideoState)

        for idx, frame in enumerate(state.frames):
            logits: list[MaskLogits] = []

            for box in state.boxes.values():
                x1, y1, x2, y2 = box.astype(int)

                data = np.full((1, *frame.shape[:2]), -1, dtype=np.float32)

                data[:, y1 : y2 + 1, x1 : x2 + 1] = 1

                if idx == 0:
                    data[:, y1 + 40 : y1 + 56, x1 + 40 : x1 + 56] = 0

                logits.append(LogitsTensor(data))

            yield idx, list(state.boxes), logits


class InterruptedSam2(RecordedSam2):
    def propagate_in_video(
        self, state: object
    ) -> Iterator[tuple[int, list[int], list[MaskLogits]]]:
        yield next(super().propagate_in_video(state))


def read_image(path: Path) -> Image:
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)

    if image is None or image.dtype != np.uint8:
        raise RuntimeError(f"Unreadable experiment image: {path}")

    return image.astype(np.uint8, copy=False)


def read_metrics(output: Path) -> list[dict[str, str]]:
    with (output / "metrics.csv").open() as stream:
        return list(csv.DictReader(stream))


@dataclass
class Scenario:
    project: Project

    def run(self, output: Path) -> list[dict[str, str]]:
        experiment_002.run(
            self.project,
            output,
            predictor_factory=lambda checkpoint: SAM2Predictor(RecordedSam2()),
        )

        return read_metrics(output)

    def selections(self) -> list[TrackSelection]:
        return [
            selection
            for individual in self.project.metadata.read_individuals()
            for selection in individual.tracks
        ]

    def artifacts(self) -> dict[str, list[Path]]:
        root = self.project.media.paths.root

        return {
            "frames": sorted((root / "media/frames").rglob("*.jpg")),
            "tracks": sorted((root / "metadata/tracks").rglob("*.json")),
            "masked_crops": sorted((root / "media/masked_crops").rglob("*.png")),
            "alignments": sorted((root / "metadata/alignment").rglob("*.json")),
            "composites": sorted((root / "media/alignment").rglob("composite.png")),
            "enhanced_composites": sorted(
                (root / "media/alignment").rglob("enhanced_composite.png")
            ),
        }


def create_scenario(root: Path, *, blank_last_frame: bool = False) -> Scenario:
    root.mkdir(parents=True)

    project = Project.open(root, manifest=root / "selections.json")

    random = np.random.default_rng(7)

    texture = random.integers(0, 256, (180, 440, 3), dtype=np.uint8)

    frames = []

    for color in [(20, 40, 60), (100, 120, 140), (220, 240, 250)]:
        image = texture.copy()

        image[48:64, 48:64] = color

        image[48:64, 272:288] = color

        frames.append(image)

    if blank_last_frame:
        frames[-1][:] = 0

    videos = []

    for video_id in ("a", "b"):
        path = root / f"{video_id}.avi"

        writer = cv.VideoWriter(str(path), cv.VideoWriter.fourcc(*"MJPG"), 5, (440, 180))

        if not writer.isOpened():
            raise RuntimeError("Could not encode experiment video")

        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

        videos.append(Video(video_id, False, path.name, str(path), len(frames)))

    individuals = [
        Individual(
            "first",
            [
                TrackSelection("a", "0", 0, 2, 0),
                TrackSelection("a", "0", 1, 2, 1),
                TrackSelection("b", "0", 0, 2, 0),
            ],
        ),
        Individual("second", [TrackSelection("a", "1", 0, 2, 1)]),
    ]

    write_json_atomic(project.metadata.paths.videos(), [asdict(video) for video in videos])

    write_json_atomic(
        project.metadata.paths.bboxes(),
        {
            "a": {"0": [8, 8, 207, 167], "1": [232, 8, 431, 167]},
            "b": {"0": [8, 8, 207, 167]},
        },
    )

    write_json_atomic(
        project.metadata.individuals_path, [asdict(individual) for individual in individuals]
    )

    return Scenario(project)
