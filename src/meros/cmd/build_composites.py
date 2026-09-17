import numpy as np

from src.meros import media, metadata
from src.meros.domain import IndividualTrack


def run() -> None:
    for individual in metadata.read_individuals():
        for track in individual.tracks:
            count = build_composite(track)

            print(
                f"{track.video_id}/{track.track_id}: "
                f"combined {count} crops."
            )


def median_composite(
    crops: list[np.ndarray],
) -> np.ndarray:
    if not crops:
        raise ValueError(
            "At least one crop is required."
        )

    shape = crops[0].shape

    if any(
        crop.shape != shape
        or crop.ndim != 3
        or crop.shape[2] != 4
        for crop in crops
    ):
        raise ValueError(
            "Aligned crops must have identical BGRA dimensions."
        )

    stack = np.stack(crops)

    valid = stack[:, :, :, 3] == 255
    covered = valid.any(axis=0)

    composite = np.zeros(
        shape,
        dtype=np.uint8,
    )

    colors = stack[
        :,
        covered,
        :3,
    ].astype(np.float32)

    colors[~valid[:, covered]] = np.nan

    composite[covered, :3] = np.rint(
        np.nanmedian(
            colors,
            axis=0,
        )
    ).astype(np.uint8)

    composite[covered, 3] = 255

    return composite


def build_composite(
    track: IndividualTrack,
) -> int:
    alignment = metadata.read_alignment(track.id)

    if (
        alignment.start_frame != track.start_frame
        or alignment.end_frame != track.end_frame
        or alignment.reference_frame
        != track.reference_frame
    ):
        raise ValueError(
            f"{track.video_id}/{track.track_id}: "
            "selection changed; rerun alignment first."
        )

    crops = [
        media.read_masked_crop(
            track.video_id,
            track.track_id,
            track.reference_frame,
        )
    ]

    seen = {track.reference_frame}

    for row in alignment.frames:
        frame_idx = row.frame_idx

        if frame_idx is None:
            raise ValueError(
                "Alignment row has no frame index."
            )

        if (
            frame_idx in seen
            or not (
                track.start_frame
                <= frame_idx
                <= track.end_frame
            )
        ):
            raise ValueError(
                f"{track.video_id}/{track.track_id}: "
                f"invalid alignment frame {frame_idx}."
            )

        seen.add(frame_idx)

        if row.status != "candidate":
            continue

        crops.append(
            media.read_aligned_crop(
                track.video_id,
                track.track_id,
                frame_idx,
            )
        )

    composite = median_composite(crops)

    media.write_composite(
        track.video_id,
        track.track_id,
        track.reference_frame,
        composite,
    )

    return len(crops)
