from __future__ import annotations

import cv2 as cv
import numpy as np

from collections.abc import Callable
from pathlib import Path
from meros.project import Project

from meros.domain.prediction import VideoPredictor
from meros.external import build_sam2_predictor
from meros.processing.extract_tracks import tracks_complete


def masked_crops_complete(project: Project) -> bool:
    if not tracks_complete(project):
        return False

    for video in project.metadata.read_videos():
        track_metadata = project.metadata.read_track(video.video_id)

        for track_id, track in track_metadata.tracks.items():
            for frame_idx, observation in track.frames.items():
                x1, y1, x2, y2 = observation.bbox

                shape = (y2 - y1 + 1, x2 - x1 + 1)

                path = project.media.paths.masked_crop(video.video_id, track_id, int(frame_idx))

                if not project.media.valid_image(path, 4, shape, mask_area=observation.mask_area):
                    return False

    return True


def run(
    project: Project,
    *,
    predictor_factory: Callable[[Path], VideoPredictor] = build_sam2_predictor,
) -> None:
    predictor = predictor_factory(project.checkpoint)

    count = sum(
        extract_masked_crops(video.video_id, predictor, project=project)
        for video in project.metadata.read_videos()
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

    # slice endpoints are exclusive; retain the final row and column.
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


def extract_masked_crops(video_id: str, predictor: VideoPredictor, *, project: Project) -> int:
    track_metadata = project.metadata.read_track(video_id)

    tracks = track_metadata.tracks

    if not tracks:
        raise ValueError(f"{video_id}: no saved tracks available.")

    state = predictor.init_state(
        video_path=str(project.media.frames_path(video_id)),
    )

    for track_id, track in tracks.items():
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=track.initial_frame,
            obj_id=int(track_id),
            box=np.asarray(track.initial_bbox, dtype=np.float32),
        )

    count = 0

    for prediction in predictor.propagate_in_video(state):
        frame_idx = prediction.frame_idx

        frame = project.media.read_frame(video_id, frame_idx)

        for track_id, mask in prediction.masks.items():
            track = tracks[track_id]

            if frame_idx < track.initial_frame or str(frame_idx) not in track.frames:
                continue

            crop = build_masked_crop(frame, mask)

            if crop is None:
                continue

            project.media.write_masked_crop(
                video_id,
                track_id,
                frame_idx,
                crop,
            )

            count += 1

    return count
