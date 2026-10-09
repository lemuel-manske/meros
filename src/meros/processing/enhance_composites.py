import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt

from meros.project import Project

from meros.domain import TrackSelection

CLIP_LIMIT = 1.5
STRENGTH = 0.5


def enhanced_composites_complete(project: Project) -> bool:
    try:
        individuals = project.metadata.read_individuals()
    except (OSError, ValueError, KeyError, TypeError):
        return False

    if not individuals or any(not ind.tracks for ind in individuals):
        return False

    for individual in individuals:
        for track in individual.tracks:
            reference_path = project.media.paths.masked_crop(
                track.video_id, track.track_id, track.reference_frame
            )

            if not project.media.valid_image(reference_path, 4):
                return False

            reference = project.media.read_masked_crop(
                track.video_id, track.track_id, track.reference_frame
            )

            path = project.media.paths.composite_enhanced(track)

            if not project.media.valid_image(path, 4, reference.shape[:2]):
                return False

    return True


def run(project: Project) -> None:
    for individual in project.metadata.read_individuals():
        for track in individual.tracks:
            enhance_composite(track, project=project)

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


def enhance_composite(track: TrackSelection, *, project: Project) -> None:
    composite = project.media.read_composite(track)

    enhanced = enhance_contrast(composite)

    project.media.write_composite_enhanced(track, enhanced)
