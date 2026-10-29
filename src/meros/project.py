"""Explicit storage and execution dependencies for subject photos and optional video processing."""

from dataclasses import dataclass
from pathlib import Path

from meros.adapters import LocalFsMediaStore, LocalFsMetadataStore, MediaPaths, MetadataPaths


class LocalFsMediaStoreWithPaths(LocalFsMediaStore):
    paths: MediaPaths


class LocalFsMetadataStoreWithPaths(LocalFsMetadataStore):
    paths: MetadataPaths


@dataclass(frozen=True)
class Project:
    media: LocalFsMediaStoreWithPaths
    metadata: LocalFsMetadataStoreWithPaths

    checkpoint: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt")

    @classmethod
    def open(
        cls,
        *,
        checkpoint: Path = Path("external/sam2/checkpoints/sam2.1_hiera_large.pt"),
    ) -> "Project":
        metadata = LocalFsMetadataStoreWithPaths()
        media = LocalFsMediaStoreWithPaths()

        metadata.paths = MetadataPaths()
        media.paths = MediaPaths()

        return cls(media, metadata, checkpoint=checkpoint)
