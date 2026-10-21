"""Map original video frame IDs to contiguous sequence indices for SAM2."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from meros.project import Project


@dataclass(frozen=True)
class FrameSequence:
    path: Path
    frame_ids: tuple[int, ...]

    def original(self, sequence_index: int) -> int:
        if not 0 <= sequence_index < len(self.frame_ids):
            raise ValueError(f"Invalid SAM2 sequence index: {sequence_index}")

        return self.frame_ids[sequence_index]

    def sequence_index(self, original_frame: int) -> int:
        try:
            return self.frame_ids.index(original_frame)
        except ValueError as error:
            raise ValueError(
                f"Original frame {original_frame} not in extracted sequence"
            ) from error


@contextmanager
def sam2_frame_sequence(project: Project, video_id: str) -> Generator[FrameSequence]:
    frame_ids = tuple(project.media.frame_ids(video_id))

    if not frame_ids:
        raise ValueError(f"{video_id}: no extracted frames")

    with TemporaryDirectory(prefix=f"meros-{video_id}-") as directory:
        path = Path(directory)

        for sequence_index, original_frame in enumerate(frame_ids):
            source = project.media.frame_path(video_id, original_frame).resolve()

            (path / f"{sequence_index:08d}.jpg").symlink_to(source)

        yield FrameSequence(path=path, frame_ids=frame_ids)
