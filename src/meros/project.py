"""Explicit storage and execution dependencies for processing and experiments."""

from dataclasses import dataclass, field
from pathlib import Path

from meros.adapters import LocalFsMediaStore, LocalFsMetadataStore
from meros.config import ExecutionOptions


@dataclass(frozen=True)
class Project:
    media: LocalFsMediaStore
    metadata: LocalFsMetadataStore

    options: ExecutionOptions = field(default_factory=ExecutionOptions)
    checkpoint: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt")

    @classmethod
    def open(
        cls,
        root: Path = Path("data"),
        *,
        manifest: Path | None = None,
        checkpoint: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt"),
    ) -> "Project":
        metadata = LocalFsMetadataStore(root)

        metadata.individuals_path = manifest

        return cls(LocalFsMediaStore(root), metadata, checkpoint=checkpoint)


default_project = Project.open()
