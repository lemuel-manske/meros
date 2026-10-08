from .adapters import LocalFsMediaStore, LocalFsMetadataStore
from .domain import Individual, Track, TrackMetadata, TrackObservation

media = LocalFsMediaStore()
metadata = LocalFsMetadataStore()

__all__ = ["media", "metadata", "Individual", "Track", "TrackMetadata", "TrackObservation"]
