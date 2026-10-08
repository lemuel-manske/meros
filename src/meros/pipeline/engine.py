from __future__ import annotations

import hashlib
import json

import cv2 as cv
import numpy as np
import scipy

from importlib.metadata import version, PackageNotFoundError
from functools import lru_cache, partial

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from meros.project import Project, default_project
from meros.domain.prediction import VideoPredictor

from meros.processing import (
    align_masked_crops,
    build_composites,
    enhance_composites,
    extract_frames,
    extract_masked_crops,
    extract_tracks,
)

from meros.processing.align_masked_crops import alignments_complete
from meros.processing.build_composites import composites_complete
from meros.processing.enhance_composites import enhanced_composites_complete
from meros.processing.extract_frames import frames_complete
from meros.processing.extract_masked_crops import masked_crops_complete
from meros.processing.extract_tracks import tracks_complete


_STATE_FOLDER = Path("data/.pipeline")


FingerprintFn = Callable[[], str]
RunFn = Callable[[], None]
CompleteFn = Callable[[], bool]


@dataclass(frozen=True)
class Stage:
    name: str
    run: RunFn
    is_complete: CompleteFn
    fingerprint: FingerprintFn
    depends_on: tuple[str, ...] = ()


class Pipeline:
    def __init__(
        self,
        stages: list[Stage],
        *,
        state_folder: Path = _STATE_FOLDER,
    ) -> None:
        self.state_folder = state_folder

        self.stages = {stage.name: stage for stage in stages}

        if len(self.stages) != len(stages):
            raise ValueError("Duplicate pipeline stage names")

        self._validate()

    def _validate(self) -> None:
        for stage in self.stages.values():
            for dependency in stage.depends_on:
                if dependency not in self.stages:
                    raise ValueError(f"{stage.name}: unknown dependency {dependency!r}.")

        visited: set[str] = set()

        active: set[str] = set()

        def visit(name: str) -> None:
            if name in active:
                raise ValueError(f"Cyclic pipeline dependency: {name!r}")

            if name in visited:
                return

            active.add(name)

            for dependency in self.stages[name].depends_on:
                visit(dependency)

            active.remove(name)

            visited.add(name)

        for name in self.stages:
            visit(name)

    def run(
        self,
        target: str,
        *,
        force: bool = False,
    ) -> None:
        if target not in self.stages:
            raise ValueError(f"Unknown pipeline stage: {target!r}.")

        self._run_targets((target,), force=force)

    def _run_targets(self, targets: tuple[str, ...], *, force: bool) -> None:
        visited: set[str] = set()

        def execute(name: str) -> None:
            if name in visited:
                return

            stage = self.stages[name]

            for dependency in stage.depends_on:
                execute(dependency)

            if not force and self._is_current(stage):
                print(f"{stage.name}: up to date")

                visited.add(name)

                return

            print(f"{stage.name}: running")

            stage.run()

            if not stage.is_complete():
                raise RuntimeError(f"{stage.name}: stage completed but its outputs are incomplete.")

            self._save_fingerprint(
                stage.name,
                stage.fingerprint(),
            )

            print(f"{stage.name}: complete")

            visited.add(name)

        for target in targets:
            execute(target)

    def run_all(
        self,
        *,
        force: bool = False,
    ) -> None:
        self._run_targets(tuple(self.stages), force=force)

    def _is_current(
        self,
        stage: Stage,
    ) -> bool:
        if not stage.is_complete():
            return False

        saved = self._read_fingerprint(
            stage.name,
        )

        if saved is None:
            return False

        return saved == stage.fingerprint()

    def _state_path(
        self,
        stage_name: str,
    ) -> Path:
        return self.state_folder / f"{stage_name}.json"

    def _read_fingerprint(
        self,
        stage_name: str,
    ) -> str | None:
        path = self._state_path(stage_name)

        if not path.exists():
            return None

        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

        if not isinstance(state, dict):
            return None

        saved = state.get("fingerprint")

        return saved if isinstance(saved, str) else None

    def _save_fingerprint(
        self,
        stage_name: str,
        fingerprint: str,
    ) -> None:
        path = self._state_path(stage_name)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                },
                f,
                indent=2,
            )

            f.write("\n")


def fingerprint(
    *values: object,
) -> str:
    """
    Produce a deterministic fingerprint for JSON-compatible values.
    """

    serialized = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_fingerprint(
    path: str | Path,
) -> str:
    """
    Hash file contents; size/mtime only memoize reads within this process.
    """

    path = Path(path)

    stat = path.stat()

    return content_hash(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=128)
def content_hash(path: str, size: int, mtime: int) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]

    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = None

    return fingerprint(
        {
            "opencv": cv.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch_version,
        },
        [(str(p.relative_to(root)), file_fingerprint(p)) for p in sorted(root.rglob("*.py"))],
    )


def videos_fingerprint(project: Project = default_project) -> str:
    """
    Raw video identity.

    If the raw video changes, downstream frame extraction becomes stale.
    """

    rows = []

    for video in project.metadata.read_videos():
        path = Path(video.path)

        rows.append(
            (
                video.video_id,
                video.mirrored,
                video.frame_count,
                file_fingerprint(path),
            )
        )

    return fingerprint(rows)


def tracks_fingerprint(project: Project = default_project) -> str:
    rows = []

    for video in project.metadata.read_videos():
        try:
            track_metadata = project.metadata.read_track(video.video_id)
        except FileNotFoundError:
            rows.append((video.video_id, None))

            continue

        tracks = []

        for track_id, track in sorted(track_metadata.tracks.items()):
            frames = [
                (
                    frame_idx,
                    observation.bbox,
                    observation.mask_area,
                )
                for frame_idx, observation in sorted(track.frames.items())
            ]

            tracks.append(
                (
                    track_id,
                    track.initial_frame,
                    track.initial_bbox,
                    frames,
                )
            )

        rows.append(
            (
                video.video_id,
                tracks,
            )
        )

    return fingerprint(rows)


def individuals_fingerprint(project: Project = default_project) -> str:
    rows = []

    for individual in project.metadata.read_individuals():
        rows.append(
            (
                individual.individual_id,
                [
                    (
                        track.video_id,
                        track.track_id,
                        track.start_frame,
                        track.end_frame,
                        track.reference_frame,
                    )
                    for track in individual.tracks
                ],
            )
        )

    return fingerprint(rows)


def frames_stage_fingerprint(project: Project = default_project) -> str:
    return fingerprint(
        "extract_frames:v3",
        implementation_fingerprint(),
        videos_fingerprint(project),
    )


def tracks_stage_fingerprint(project: Project = default_project) -> str:
    return fingerprint(
        "extract_tracks:v3",
        project.metadata.read_bboxes(),
        extract_tracks.AUTO,
        file_fingerprint(project.checkpoint),
        frames_stage_fingerprint(project),
    )


def masked_crops_stage_fingerprint(project: Project = default_project) -> str:
    return fingerprint(
        "extract_masked_crops:v2",
        tracks_stage_fingerprint(project),
        tracks_fingerprint(project),
        "sam2.1_hiera_large",
        file_fingerprint(project.checkpoint),
    )


def alignments_stage_fingerprint(project: Project = default_project) -> str:
    return fingerprint(
        "align_masked_crops:v2",
        # Important: alignment now depends explicitly
        # on the upstream masked-crop stage.
        masked_crops_stage_fingerprint(project),
        individuals_fingerprint(project),
        {
            "sift_features": align_masked_crops.SIFT_FEATURES,
            "ratio": align_masked_crops.MATCH_RATIO,
            "min_inliers": align_masked_crops.MIN_INLIERS,
            "ransac_error": align_masked_crops.MAX_ERROR,
            "ransac_iterations": align_masked_crops.RANSAC_MAX_ITERS,
            "ransac_confidence": align_masked_crops.RANSAC_CONFIDENCE,
            "min_inlier_ratio": align_masked_crops.MIN_INLIER_RATIO,
            "min_hull_fraction": align_masked_crops.MIN_HULL_FRACTION,
        },
    )


def composites_stage_fingerprint(project: Project = default_project) -> str:
    return fingerprint(
        "build_composites:v2",
        alignments_stage_fingerprint(project),
        individuals_fingerprint(project),
        {
            "method": "median",
        },
    )


def enhanced_composites_stage_fingerprint(project: Project = default_project) -> str:
    return fingerprint(
        "enhance_composites:v2",
        composites_stage_fingerprint(project),
        {
            "method": "clahe",
            "clip_limit": enhance_composites.CLIP_LIMIT,
            "strength": enhance_composites.STRENGTH,
            "tile_grid": [8, 8],
        },
    )


def create_pipeline(
    project: Project = default_project,
    *,
    predictor_factory: Callable[[], VideoPredictor] | None = None,
) -> Pipeline:
    """Bind stages to one project and an explicit inference implementation."""

    stages = [
        Stage(
            name="frames",
            run=partial(extract_frames.run, project=project),
            is_complete=partial(frames_complete, project=project),
            fingerprint=partial(frames_stage_fingerprint, project=project),
        ),
        Stage(
            name="tracks",
            run=partial(extract_tracks.run, project=project, predictor_factory=predictor_factory),
            is_complete=partial(tracks_complete, project=project),
            fingerprint=partial(tracks_stage_fingerprint, project=project),
            depends_on=("frames",),
        ),
        Stage(
            name="masked_crops",
            run=partial(
                extract_masked_crops.run, project=project, predictor_factory=predictor_factory
            ),
            is_complete=partial(masked_crops_complete, project=project),
            fingerprint=partial(masked_crops_stage_fingerprint, project=project),
            depends_on=("tracks",),
        ),
        Stage(
            name="alignments",
            run=partial(align_masked_crops.run, project=project),
            is_complete=partial(alignments_complete, project=project),
            fingerprint=partial(alignments_stage_fingerprint, project=project),
            depends_on=("masked_crops",),
        ),
        Stage(
            name="composites",
            run=partial(build_composites.run, project=project),
            is_complete=partial(composites_complete, project=project),
            fingerprint=partial(composites_stage_fingerprint, project=project),
            depends_on=("alignments",),
        ),
        Stage(
            name="enhanced_composites",
            run=partial(enhance_composites.run, project=project),
            is_complete=partial(enhanced_composites_complete, project=project),
            fingerprint=partial(enhanced_composites_stage_fingerprint, project=project),
            depends_on=("composites",),
        ),
    ]

    return Pipeline(stages, state_folder=project.media.paths.root / ".pipeline")
