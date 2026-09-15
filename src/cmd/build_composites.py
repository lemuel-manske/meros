import numpy as np

from src.alignment import (
    persist_composite,
    read_aligned_crop,
    read_alignment_metadata,
)
from src.artifacts import path_to_composite
from src.masked_crop import read_masked_crop

from src.meros.dataset import individuals


def median_composite(crops: list[np.ndarray]) -> np.ndarray:
    if not crops:
        raise ValueError("At least one crop is required.")

    shape = crops[0].shape

    if any(crop.shape != shape or crop.ndim != 3 or crop.shape[2] != 4 for crop in crops):
        raise ValueError("Aligned crops must have identical BGRA dimensions.")

    stack = np.stack(crops)

    # interpolated mask edges can contain background-darkened colors.
    valid = stack[:, :, :, 3] == 255
    covered = valid.any(axis=0)
    composite = np.zeros(shape, dtype=np.uint8)

    # evaluate only covered pixels to avoid all-nan background slices.
    colors = stack[:, covered, :3].astype(np.float32)
    colors[~valid[:, covered]] = np.nan
    composite[covered, :3] = np.rint(np.nanmedian(colors, axis=0)).astype(np.uint8)
    composite[covered, 3] = 255

    return composite


def build_composite(track) -> int:
    vid_id = track.video_id
    track_id = track.track_id

    metadata = read_alignment_metadata(vid_id, track_id)

    if any(metadata[key] != track[key] for key in track):
        raise ValueError(f"{vid_id}/{track_id}: selection changed; rerun alignment first.")

    reference_frame = track.reference_frame
    crops = [read_masked_crop(vid_id, track_id, reference_frame)]
    seen = {reference_frame}

    for row in metadata["frames"]:
        frame_idx = row["frame_idx"]

        if frame_idx in seen or not track.start_frame <= frame_idx <= track.end_frame:
            raise ValueError(f"{vid_id}/{track_id}: invalid alignment frame {frame_idx}.")

        seen.add(frame_idx)

        if row["status"] != "candidate":
            continue

        crops.append(read_aligned_crop(vid_id, track_id, frame_idx))

    composite = median_composite(crops)

    persist_composite(vid_id, track_id, composite)

    return len(crops)


if __name__ == "__main__":
    for individual in individuals:
        for track in individual.tracks:
            count = build_composite(track)

            path = path_to_composite(track.video_id, track.track_id)

            print(f"Combined {count} crops: {path}.")
