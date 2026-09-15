import cv2 as cv
import numpy as np

from scipy.ndimage import distance_transform_edt

from src.artifacts import (
    path_to_composite,
    path_to_composite_comparison,
    path_to_enhanced_composite,
)

from src.meros.dataset import individuals


def enhance_contrast(
    composite: np.ndarray,
    clip_limit: float = 1.5,
    strength: float = 0.5,
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

    # extend foreground brightness so transparent black does not dominate edge tiles.
    indices = np.empty((2, *valid.shape), dtype=np.int32)
    distance_transform_edt(~valid, return_distances=False, return_indices=True, indices=indices)
    filled = luminance[tuple(indices)]
    enhanced = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(filled)
    lab[:, :, 0] = cv.addWeighted(luminance, 1 - strength, enhanced, strength, 0)

    result = composite.copy()
    result[:, :, :3] = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    result[~valid, :3] = 0

    return result


def enhance_composite(vid_id: str, track_id: str) -> str:
    source = path_to_composite(vid_id, track_id)
    composite = cv.imread(source, cv.IMREAD_UNCHANGED)

    if composite is None:
        raise ValueError(f"Could not read composite: {source}.")

    enhanced = enhance_contrast(composite)
    output = path_to_enhanced_composite(vid_id, track_id)

    if not cv.imwrite(output, enhanced):
        raise RuntimeError(f"Could not save {output}.")

    comparison = np.concatenate((composite, enhanced), axis=1)
    output = path_to_composite_comparison(vid_id, track_id)

    if not cv.imwrite(output, comparison):
        raise RuntimeError(f"Could not save {output}.")

    return output


if __name__ == "__main__":
    for individual in individuals:
        for track in individual.tracks:
            path = enhance_composite(track.video_id, track.track_id)

            print(f"Original left, enhanced right: {path}.")
