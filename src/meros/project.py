"""Explicit storage and execution dependencies for processing and experiments."""

from dataclasses import dataclass
from pathlib import Path

from meros.adapters import LocalFsMediaStore, LocalFsMetadataStore


@dataclass(frozen=True)
class Project:
    media: LocalFsMediaStore
    metadata: LocalFsMetadataStore

    checkpoint: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt")

    @classmethod
    def open(
        cls,
        root: Path = Path("data"),
        *,
        manifest: Path,
        checkpoint: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt"),
    ) -> "Project":
        metadata = LocalFsMetadataStore(root, manifest)

        return cls(LocalFsMediaStore(root), metadata, checkpoint=checkpoint)
