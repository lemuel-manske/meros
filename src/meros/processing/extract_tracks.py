from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2 as cv
import numpy as np

from meros.domain import BBox, Track, TrackMetadata, TrackObservation
from meros.domain.prediction import Prediction, VideoPredictor
from meros.external import build_sam2_predictor
from meros.project import Project


def run(
    project: Project,
    *,
    predictor_factory: Callable[[Path], VideoPredictor] = build_sam2_predictor,
) -> None:
    predictor = predictor_factory(project.checkpoint)

    for video in project.metadata.read_videos():
        video_id = video.video_id

        tracks = select_track_seeds(video_id, project=project)

        track_metadata = propagate_tracks(video_id, tracks, predictor, project=project)

        if not track_metadata.tracks or any(not t.frames for t in track_metadata.tracks.values()):
            raise RuntimeError(f"{video_id}: incomplete tracking; saved metadata was not replaced")

        project.metadata.write_track(video_id, track_metadata)

        print(f"Saved {len(tracks)} tracks for {video_id}.")


def tracks_complete(project: Project) -> bool:
    try:
        videos = project.metadata.read_videos()

        seeds = project.metadata.read_bboxes()
    except (OSError, ValueError, KeyError, TypeError):
        return False

    if not videos:
        return False

    for video in videos:
        try:
            saved = project.metadata.read_track(video.video_id)

            frame_ids = project.media.frame_ids(video.video_id)
        except (OSError, ValueError, KeyError, TypeError):
            return False

        if (
            not frame_ids
            or frame_ids != list(range(len(frame_ids)))
            or not saved.tracks
            or saved.video_id != video.video_id
            or saved.processed_frame_count != len(frame_ids)
        ):
            return False

        try:
            height, width = project.media.read_frame(video.video_id, 0).shape[:2]
        except (OSError, ValueError, cv.error):
            return False

        video_seeds = seeds.get(video.video_id, {})

        if set(saved.tracks) != set(video_seeds):
            return False

        for track_id, track in saved.tracks.items():
            if not track.frames:
                return False

            if track.initial_frame != 0 or track.initial_bbox != video_seeds[track_id]:
                return False

            try:
                for idx, observation in track.frames.items():
                    if str(int(idx)) != idx or not track.initial_frame <= int(idx) < len(frame_ids):
                        return False

                    x1, y1, x2, y2 = observation.bbox

                    if not 0 <= x1 <= x2 < width or not 0 <= y1 <= y2 < height:
                        return False
            except (ValueError, TypeError):
                return False

    return True


def select_fish_bboxes(video_id: str, frame: np.ndarray) -> BBox:
    window = f"{video_id}: Enter = save, a = add fish, Esc = cancel"

    boxes: BBox = {}

    def add_fish() -> None:
        x, y, width, height = cv.selectROI(window, frame, showCrosshair=True, fromCenter=False)

        if width > 0 and height > 0:
            boxes[str(len(boxes))] = [int(x), int(y), int(x + width - 1), int(y + height - 1)]

        elif not boxes:
            raise ValueError(f"{video_id}: no fish selected; seed boxes were not saved")

    print(f"{video_id}: draw a fish box and press Enter/Space to confirm.")

    print("Then press Enter to save, a to add another fish, or Esc to cancel without saving.")

    cv.namedWindow(window, cv.WINDOW_NORMAL)

    try:
        add_fish()

        while cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) >= 1:
            preview = frame.copy()

            for track_id, (x1, y1, x2, y2) in boxes.items():
                cv.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv.putText(
                    preview,
                    track_id,
                    (x1, max(20, y1)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv.imshow(window, preview)

            key = cv.waitKey(30) & 0xFF

            if key in (10, 13):
                break

            if key == 27:
                raise ValueError(f"{video_id}: fish selection cancelled; seed boxes were not saved")

            if key == ord("a"):
                add_fish()
        else:
            raise ValueError(f"{video_id}: selection window closed; seed boxes were not saved")
    finally:
        cv.destroyWindow(window)

    if not boxes:
        raise ValueError(f"{video_id}: no fish selected; seed boxes were not saved")

    return boxes


def select_track_seeds(
    video_id: str,
    *,
    project: Project,
    bbox_selector: Callable[[str, np.ndarray], BBox] = select_fish_bboxes,
) -> dict[str, Track]:
    saved_boxes = project.metadata.read_bboxes()

    boxes = saved_boxes.get(video_id)

    if not boxes:
        frame = project.media.read_frame(video_id, 0)

        boxes = bbox_selector(video_id, frame)

        saved_boxes[video_id] = boxes

        project.metadata.write_bboxes(saved_boxes)

    return {
        track_id: Track(initial_frame=0, initial_bbox=bbox, frames={})
        for track_id, bbox in boxes.items()
    }


def propagate_tracks(
    video_id: str,
    tracks: dict[str, Track],
    predictor: VideoPredictor,
    *,
    project: Project,
) -> TrackMetadata:

    frame_ids = project.media.frame_ids(video_id)

    if not frame_ids or frame_ids != list(range(len(frame_ids))):
        raise ValueError(f"{video_id}: expected contiguous zero-based frames")

    track_metadata = TrackMetadata(video_id, tracks, len(frame_ids))

    state = predictor.init_state(str(project.media.frames_path(video_id)))

    for track_id, track in tracks.items():
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=track.initial_frame,
            obj_id=int(track_id),
            box=np.asarray(track.initial_bbox, dtype=np.float32),
        )

    processed: set[int] = set()

    for prediction in predictor.propagate_in_video(state):
        if prediction.frame_idx in processed or not 0 <= prediction.frame_idx < len(frame_ids):
            raise RuntimeError("Invalid predictor frame index")

        processed.add(prediction.frame_idx)

        process_frame(video_id, prediction, track_metadata, project=project)

    if processed != set(frame_ids):
        raise RuntimeError("SAM2 did not process the full sequence")

    return track_metadata


def process_frame(
    video_id: str,
    prediction: Prediction,
    track_metadata: TrackMetadata,
    *,
    project: Project,
) -> None:
    frame_idx = prediction.frame_idx

    frame = project.media.read_frame(
        video_id,
        frame_idx,
    )

    for track_id, mask in prediction.masks.items():
        track = track_metadata.tracks[track_id]

        if mask.shape != frame.shape[:2]:
            raise ValueError(f"{video_id}/{frame_idx}: mask and frame dimensions differ")

        ys, xs = np.where(mask)

        if not len(xs):
            continue

        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        track.frames[str(frame_idx)] = TrackObservation(
            bbox=bbox,
            mask_area=int(mask.sum()),
        )
