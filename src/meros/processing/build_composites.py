import numpy as np
import cv2 as cv
from meros import media, metadata
from meros.domain import TrackSelection


def composites_complete() -> bool:
    individuals = metadata.read_individuals()
    if not individuals or not any(ind.tracks for ind in individuals):
        return False
    return all(
        (
            media.composite_exists(
                track.video_id,
                track.track_id,
                track.reference_frame,
                selection_id=track.selection_id,
            )
            for individual in individuals
            for track in individual.tracks
        )
    )


def run() -> None:
    for individual in metadata.read_individuals():
        for track in individual.tracks:
            count = build_composite(track)
            print(f"{track.video_id}/{track.track_id}: combined {count} crops.")


def median_composite(crops: list[np.ndarray]) -> np.ndarray:
    if not crops:
        raise ValueError("At least one crop is required.")
    shape = crops[0].shape
    if any((crop.shape != shape or crop.ndim != 3 or crop.shape[2] != 4 for crop in crops)):
        raise ValueError("Aligned crops must have identical BGRA dimensions.")
    stack = np.stack(crops)
    valid = stack[:, :, :, 3] == 255
    covered = valid.any(axis=0)
    composite = np.zeros(shape, dtype=np.uint8)
    colors = stack[:, covered, :3].astype(np.float32)
    colors[~valid[:, covered]] = np.nan
    composite[covered, :3] = np.rint(np.nanmedian(colors, axis=0)).astype(np.uint8)
    composite[covered, 3] = 255
    return composite


def build_composite(track: TrackSelection) -> int:
    alignment = metadata.read_alignment(track.id)
    if (
        alignment.start_frame != track.start_frame
        or alignment.end_frame != track.end_frame
        or alignment.reference_frame != track.reference_frame
    ):
        raise ValueError(
            f"{track.video_id}/{track.track_id}: selection changed; rerun alignment first."
        )
    crops = [media.read_masked_crop(track.video_id, track.track_id, track.reference_frame)]
    seen = {track.reference_frame}
    for row in alignment.frames:
        frame_idx = row.frame_idx
        if frame_idx is None:
            raise ValueError("Alignment row has no frame index.")
        if frame_idx in seen or not track.start_frame <= frame_idx <= track.end_frame:
            raise ValueError(
                f"{track.video_id}/{track.track_id}: invalid alignment frame {frame_idx}."
            )
        seen.add(frame_idx)
        if row.status != "accepted":
            continue
        if row.source_to_reference is None:
            raise ValueError("Accepted alignment has no transform")
        source = media.read_masked_crop(track.video_id, track.track_id, frame_idx)
        crops.append(
            cv.warpAffine(
                source, np.asarray(row.source_to_reference), (crops[0].shape[1], crops[0].shape[0])
            )
        )
    composite = median_composite(crops)
    media.write_composite(
        track.video_id,
        track.track_id,
        track.reference_frame,
        composite,
        selection_id=track.selection_id,
    )
    return len(crops)
