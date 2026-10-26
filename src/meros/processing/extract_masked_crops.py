from __future__ import annotations

import cv2 as cv
import numpy as np

from collections.abc import Callable
from meros.project import Project
from pathlib import Path

from meros.domain.prediction import VideoPredictor
from meros.external import build_sam2_predictor
from meros.processing.extract_tracks import tracks_complete
from meros.processing.frame_sequence import frame_batches, sam2_frame_sequence


def video_masked_crops_complete(project: Project, video_id: str) -> bool:
    try:
        track_metadata = project.metadata.read_track(video_id)
    except (OSError, ValueError, KeyError, TypeError):
        return False

    if not track_metadata.tracks:
        return False

    for track_id, track in track_metadata.tracks.items():
        for frame_idx, observation in track.frames.items():
            x1, y1, x2, y2 = observation.bbox

            shape = (y2 - y1 + 1, x2 - x1 + 1)

            path = project.media.paths.masked_crop(video_id, track_id, int(frame_idx))

            if not project.media.valid_image(path, 4, shape, mask_area=observation.mask_area):
                return False

    return True


def masked_crops_complete(project: Project) -> bool:
    if not tracks_complete(project):
        return False

    return all(
        video_masked_crops_complete(project, video.video_id)
        for video in project.metadata.read_videos()
    )


def batch_masked_crops_complete(
    project: Project,
    video_id: str,
    tracks: dict,
    batch: list[int],
) -> bool:
    for frame_idx in batch:
        for track_id, track in tracks.items():
            observation = track.frames.get(str(frame_idx))

            if observation is None:
                continue

            x1, y1, x2, y2 = observation.bbox
            shape = (y2 - y1 + 1, x2 - x1 + 1)

            path = project.media.paths.masked_crop(video_id, track_id, frame_idx)

            if not project.media.valid_image(path, 4, shape, mask_area=observation.mask_area):
                return False

    return True


def run(
    project: Project,
    *,
    predictor_factory: Callable[[Path], VideoPredictor] = build_sam2_predictor,
) -> None:
    predictor = None
    count = 0

    for video in project.metadata.read_videos():
        video_id = video.video_id

        if video_masked_crops_complete(project, video_id):
            print(f"{video_id}: masked crops already complete, skipping.")
            continue

        if predictor is None:
            predictor = predictor_factory(project.checkpoint)

        count += extract_masked_crops(
            video_id,
            predictor,
            project=project,
        )

    print(f"Saved {count} new masked crops.")


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

    count = 0

    frame_ids = tuple(project.media.frame_ids(video_id))

    written_frames: set[int] = set()

    for batch in frame_batches(frame_ids):
        if batch_masked_crops_complete(project, video_id, tracks, list(batch)):
            print(f"{video_id}: batch already complete, skipping.")
            continue

        with sam2_frame_sequence(project, video_id, batch) as sequence:
            state = predictor.init_state(video_path=str(sequence.path))

            for track_id, track in tracks.items():
                first_observation = track.frames.get(str(batch[0]))

                if first_observation is None:
                    continue

                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(track_id),
                    box=np.asarray(first_observation.bbox, dtype=np.float32),
                )

            for prediction in predictor.propagate_in_video(state):
                frame_idx = sequence.original(prediction.frame_idx)

                if frame_idx in written_frames:
                    continue

                written_frames.add(frame_idx)

                frame = project.media.read_frame(video_id, frame_idx)

                for track_id, mask in prediction.masks.items():
                    track = tracks[track_id]

                    observation = track.frames.get(str(frame_idx))

                    if observation is None:
                        continue

                    # Skip existing valid crops before building them
                    x1, y1, x2, y2 = observation.bbox
                    shape = (y2 - y1 + 1, x2 - x1 + 1)

                    path = project.media.paths.masked_crop(video_id, track_id, frame_idx)

                    if project.media.valid_image(
                        path,
                        4,
                        shape,
                        mask_area=observation.mask_area,
                    ):
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

            del state

        print(f"{video_id}: saved {count} crops.", flush=True)

    return count


if __name__ == "__main__":
    run(Project.open())
