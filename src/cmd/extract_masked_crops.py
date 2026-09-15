import cv2 as cv
import numpy as np
import torch

from src.artifacts import (
    path_to_frame,
    path_to_frames,
    path_to_masked_crop,
)
from src.metadata import get_track_metadata

from src.meros.dataset import videos
from src.meros.external import build_sam2_predictor, SAM2Predictor



def build_masked_crop(
    frame: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    if mask.shape != frame.shape[:2]:
        raise ValueError("Mask and frame dimensions differ.")

    mask = mask.astype(bool)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    # slice endpoints are exclusive; retain the final row and column of the fish.
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    crop_mask = mask[y1:y2, x1:x2]

    crop = cv.cvtColor(frame[y1:y2, x1:x2], cv.COLOR_BGR2BGRA)

    crop[~crop_mask] = 0

    crop[:, :, 3] = crop_mask.astype(np.uint8) * 255

    return crop


def extract_masked_crops(
    vid_id: str,
    predictor: SAM2Predictor,
) -> int:
    metadata = get_track_metadata(vid_id)
    tracks = metadata["tracks"]

    if not tracks:
        raise ValueError(f"{vid_id}: no saved tracks available.")

    state = predictor.init_state(
        video_path=path_to_frames(vid_id),
    )

    for track_id, track in tracks.items():
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=track["initial_frame"],
            obj_id=int(track_id),
            box=np.asarray(track["initial_bbox"], dtype=np.float32),
        )

    count = 0

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        frame = cv.imread(path_to_frame(vid_id, frame_idx))

        if frame is None:
            raise RuntimeError(f"{vid_id}: could not read frame {frame_idx}.")

        for obj_id, logits in zip(obj_ids, mask_logits):
            track_id = str(int(obj_id))

            track = tracks[track_id]

            # only export observations belonging to the existing track metadata.
            if frame_idx < track["initial_frame"] or str(frame_idx) not in track["frames"]:
                continue

            mask = (logits > 0).cpu().numpy()

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]

            crop = build_masked_crop(frame, mask)

            if crop is None:
                continue

            output = path_to_masked_crop(vid_id, frame_idx, track_id)

            if not cv.imwrite(output, crop):
                raise RuntimeError(f"Could not save masked crop: {output}.")

            count += 1

    return count


if __name__ == "__main__":
    """
    Extract masked crops for all videos in the dataset.
    """
    predictor = build_sam2_predictor()

    with torch.inference_mode():
        count = sum(extract_masked_crops(vid.video_id, predictor) for vid in videos)

    print(f"Saved {count} masked crops.")
