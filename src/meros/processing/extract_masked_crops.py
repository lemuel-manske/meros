from __future__ import annotations

import cv2 as cv
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meros.external import SAM2Predictor

from meros import media, metadata


def masked_crops_complete() -> bool:
    for video in metadata.read_videos():
        try:
            track_metadata = metadata.read_track(video.video_id)
        except FileNotFoundError:
            return False

        for (
            track_id,
            track,
        ) in track_metadata.tracks.items():
            for frame_idx in track.frames:
                if not media.masked_crop_exists(
                    video.video_id,
                    track_id,
                    int(frame_idx),
                ):
                    return False

    return True


def run() -> None:
    import torch
    from meros.external import build_sam2_predictor

    predictor = build_sam2_predictor()

    with torch.inference_mode():
        count = sum(
            extract_masked_crops(video.video_id, predictor) for video in metadata.read_videos()
        )

    print(f"Saved {count} masked crops.")


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

    # Slice endpoints are exclusive; retain the final row and column.
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    y1, y2 = int(ys.min()), int(ys.max()) + 1

    crop_mask = mask[y1:y2, x1:x2]

    crop = cv.cvtColor(
        frame[y1:y2, x1:x2],
        cv.COLOR_BGR2BGRA,
    )

    crop[~crop_mask] = 0

    crop[:, :, 3] = crop_mask.astype(np.uint8) * 255

    return crop


def extract_masked_crops(
    video_id: str,
    predictor: SAM2Predictor,
) -> int:
    track_metadata = metadata.read_track(video_id)

    tracks = track_metadata.tracks

    if not tracks:
        raise ValueError(f"{video_id}: no saved tracks available.")

    state = predictor.init_state(
        video_path=str(media.frames_path(video_id)),
    )

    for track_id, track in tracks.items():
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=track.initial_frame,
            obj_id=int(track_id),
            box=np.asarray(track.initial_bbox, dtype=np.float32),
        )

    count = 0

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        frame = media.read_frame(video_id, frame_idx)

        for obj_id, logits in zip(obj_ids, mask_logits):
            track_id = str(int(obj_id))

            track = tracks[track_id]

            # Only export observations belonging to the saved track.
            if frame_idx < track.initial_frame or str(frame_idx) not in track.frames:
                continue

            mask = (logits > 0).cpu().numpy()

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]

            crop = build_masked_crop(frame, mask)

            if crop is None:
                continue

            media.write_masked_crop(
                video_id,
                track_id,
                frame_idx,
                crop,
            )

            count += 1

    return count
