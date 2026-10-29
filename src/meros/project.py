"""Explicit storage and execution dependencies for subject photos and optional video processing."""

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
        *,
        checkpoint: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt"),
    ) -> "Project":
        metadata = LocalFsMetadataStore()
        media = LocalFsMediaStore()

        return cls(media, metadata, checkpoint=checkpoint)
