import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt

from meros import media, metadata
from meros.domain import TrackSelection
from meros.config import options

CLIP_LIMIT = 1.5
STRENGTH = 0.5


def enhanced_composites_complete() -> bool:
    individuals = metadata.read_individuals()

    if not individuals or not any(ind.tracks for ind in individuals):
        return False

    return all(
        (
            media.composite_enhanced_exists(
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
            enhance_composite(track)

            print(f"{track.video_id}/{track.track_id}: enhanced composite complete.")


def enhance_contrast(
    composite: np.ndarray, clip_limit: float = CLIP_LIMIT, strength: float = STRENGTH
) -> np.ndarray:
    if composite.dtype != np.uint8 or composite.ndim != 3 or composite.shape[2] != 4:
        raise ValueError("Expected a uint8 BGRA composite.")

    if clip_limit <= 0 or not 0 <= strength <= 1:
        raise ValueError("Clip limit must be positive and strength between zero and one.")

    valid = composite[:, :, 3] > 0

    if not valid.any() or strength == 0:
        return composite.copy()

    lab = cv.cvtColor(composite[:, :, :3], cv.COLOR_BGR2LAB)

    luminance = lab[:, :, 0]

    indices = np.empty((2, *valid.shape), dtype=np.int32)

    distance_transform_edt(~valid, return_distances=False, return_indices=True, indices=indices)

    filled = luminance[tuple(indices)]

    enhanced = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(filled)

    lab[:, :, 0] = cv.addWeighted(luminance, 1 - strength, enhanced, strength, 0)

    result = composite.copy()

    result[:, :, :3] = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    result[~valid, :3] = 0

    return result


def enhance_composite(track: TrackSelection) -> None:
    video_id, track_id, reference_frame = (track.video_id, track.track_id, track.reference_frame)

    composite = media.read_composite(
        video_id, track_id, reference_frame, selection_id=track.selection_id
    )

    enhanced = enhance_contrast(composite)

    media.write_composite_enhanced(
        video_id, track_id, reference_frame, enhanced, selection_id=track.selection_id
    )

    if options.diagnostics:
        comparison = np.concatenate((composite, enhanced), axis=1)

        media.write_composite_comparison(
            video_id, track_id, reference_frame, comparison, selection_id=track.selection_id
        )
