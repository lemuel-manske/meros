"""Run preparation in order, rebuilding downstream outputs when a step changes."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from meros.domain import VideoPredictor
from meros.external import build_sam2_predictor
from meros.project import Project
from meros.processing import (
    extract_frames,
    extract_masked_crops,
    extract_tracks,
)


@dataclass(frozen=True)
class Stage:
    name: str

    run: Callable[[], None]

    is_complete: Callable[[], bool]


class Pipeline:
    def __init__(self, stages: list[Stage]) -> None:
        self.stages = stages

    def run(self) -> None:
        rebuild = False

        for stage in self.stages:
            if not rebuild and stage.is_complete():
                print(f"{stage.name}: complete")

                continue

            stage.run()

            if not stage.is_complete():
                raise RuntimeError(f"{stage.name}: stage completed but its outputs are incomplete.")

            rebuild = True

            print(f"{stage.name}: complete")


def create_pipeline(
    project: Project,
    *,
    predictor_factory: Callable[[Path], VideoPredictor] = build_sam2_predictor,
) -> Pipeline:
    return Pipeline(
        [
            Stage(
                "frames",
                partial(extract_frames.run, project),
                partial(extract_frames.frames_complete, project),
            ),
            Stage(
                "tracks",
                partial(extract_tracks.run, project, predictor_factory=predictor_factory),
                partial(extract_tracks.tracks_complete, project),
            ),
            Stage(
                "masked_crops",
                partial(extract_masked_crops.run, project, predictor_factory=predictor_factory),
                partial(extract_masked_crops.masked_crops_complete, project),
            ),
        ]
    )


if __name__ == "__main__":
    project = Project.open()

    pipeline = create_pipeline(project)

    pipeline.run()
