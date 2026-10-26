from __future__ import annotations

import cv2 as cv
import numpy as np

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from meros.domain import BBox, Track, TrackMetadata, TrackObservation
from meros.domain.prediction import Prediction, VideoPredictor
from meros.external import build_sam2_predictor
from meros.processing.frame_sequence import frame_batches, sam2_frame_sequence
from meros.project import Project


def video_tracks_complete(
    project: Project,
    video_id: str,
    seeds: dict,
) -> bool:
    print(f"{video_id}: checking if tracks are complete...")
    try:
        saved = project.metadata.read_track(video_id)
        frame_ids = project.media.frame_ids(video_id)
    except (OSError, ValueError, KeyError, TypeError):
        return False

    if (
        not frame_ids
        or not saved.tracks
        or saved.video_id != video_id
        or saved.processed_frame_count != len(frame_ids)
    ):
        return False

    try:
        height, width = project.media.read_frame(video_id, frame_ids[0]).shape[:2]
    except (OSError, ValueError, cv.error):
        return False

    video_seeds = seeds.get(video_id, {})

    if set(saved.tracks) != set(video_seeds):
        return False

    for track_id, track in saved.tracks.items():
        if not track.frames:
            return False

        if track.initial_frame != frame_ids[0] or track.initial_bbox != video_seeds[track_id]:
            return False

        try:
            for idx, observation in track.frames.items():
                if str(int(idx)) != idx or int(idx) not in frame_ids:
                    return False

                x1, y1, x2, y2 = observation.bbox

                if not 0 <= x1 <= x2 < width or not 0 <= y1 <= y2 < height:
                    return False

        except (ValueError, TypeError):
            return False

    return True


def tracks_complete(project: Project) -> bool:
    try:
        videos = project.metadata.read_videos()
        seeds = project.metadata.read_bboxes()
    except (OSError, ValueError, KeyError, TypeError):
        return False

    if not videos:
        return False

    return all(video_tracks_complete(project, video.video_id, seeds) for video in videos)


def run(
    project: Project,
    *,
    predictor_factory: Callable[[Path], VideoPredictor] = build_sam2_predictor,
) -> None:
    predictor = None

    for video in project.metadata.read_videos():
        video_id = video.video_id

        # Reload seeds to avoid stale metadata
        seeds = project.metadata.read_bboxes()

        if video_tracks_complete(project, video_id, seeds):
            print(f"{video_id}: already tracked, skipping.")
            continue

        if predictor is None:
            predictor = predictor_factory(project.checkpoint)

        tracks = select_track_seeds(video_id, project=project)

        track_metadata = propagate_tracks(video_id, tracks, predictor, project=project)

        if not track_metadata.tracks or any(not t.frames for t in track_metadata.tracks.values()):
            raise RuntimeError(f"{video_id}: incomplete tracking; saved metadata was not replaced")

        project.metadata.write_track(video_id, track_metadata)

        print(f"Saved {len(tracks)} tracks for {video_id}.")


@dataclass
class BoxSelectionState:
    adding: bool = False

    drag_start: tuple[int, int] | None = None

    pending: list[int] | None = None


# meant for manual selection of fish bounding boxes in the first frame of a video, to be used as seeds for tracking
def select_fish_bboxes(video_id: str, frame: np.ndarray) -> BBox:
    window = f"{video_id}: a = add fish, Enter = confirm box, q = save, Esc = cancel"

    boxes: BBox = {}

    height, width = frame.shape[:2]

    state = BoxSelectionState()

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        _ = (_flags, _userdata)

        if not state.adding:
            return

        x = min(max(x, 0), width - 1)

        y = min(max(y, 0), height - 1)

        if event == cv.EVENT_LBUTTONDOWN:
            state.drag_start = (x, y)

            state.pending = None

        elif state.drag_start is not None and event in (cv.EVENT_MOUSEMOVE, cv.EVENT_LBUTTONUP):
            start_x, start_y = state.drag_start

            state.pending = [min(start_x, x), min(start_y, y), max(start_x, x), max(start_y, y)]

            if event == cv.EVENT_LBUTTONUP:
                state.drag_start = None

    print(f"{video_id}: press a, drag a fish box, then press Enter/Space to confirm.")

    print("Press a for another fish, q to save and continue, or Esc to cancel without saving.")

    cv.namedWindow(window, cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)

    try:
        cv.imshow(window, frame)

        cv.setMouseCallback(window, on_mouse)

        while True:
            preview = frame.copy()

            for track_id, (x1, y1, x2, y2) in boxes.items():
                cv.rectangle(preview, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                cv.putText(
                    preview,
                    track_id,
                    (x1, max(20, y1)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 255, 0),
                    thickness=2,
                )

            if state.pending is not None:
                x1, y1, x2, y2 = state.pending

                cv.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv.imshow(window, preview)

            key = cv.waitKeyEx(30)

            try:
                visible = cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) >= 1
            except cv.error:
                visible = False

            if not visible or key in (27, 0x01000000):
                raise ValueError(f"{video_id}: fish selection cancelled; seed boxes were not saved")

            if key == ord("a"):
                state.adding = True

                state.drag_start = None

                state.pending = None

            elif key in (13, 10, 32):  # Enter, Space
                if state.adding and state.pending is not None:
                    x1, y1, x2, y2 = state.pending

                    if x1 == x2 or y1 == y2:
                        print(f"{video_id}: box must have nonzero width and height.")

                        continue

                    track_id = str(len(boxes))

                    boxes[track_id] = state.pending.copy()

                    state.adding = False
                    state.drag_start = None
                    state.pending = None

            elif key == ord("q"):
                if not boxes:
                    raise ValueError(f"{video_id}: no fish selected; seed boxes were not saved")

                return boxes
    finally:
        try:
            cv.destroyWindow(window)
        except cv.error:
            pass


def select_track_seeds(
    video_id: str,
    *,
    project: Project,
) -> dict[str, Track]:
    bboxes = project.metadata.read_bboxes()
    box = bboxes.get(video_id)

    first_frame = project.media.frame_ids(video_id)[0]

    # if the bounding boxes for this video have not been selected yet, prompt the user to select them
    if not box:
        frame = project.media.read_frame(video_id, first_frame)

        box = select_fish_bboxes(video_id, frame)

        bboxes[video_id] = box

        project.metadata.write_bboxes(bboxes)

    return {
        track_id: Track(initial_frame=first_frame, initial_bbox=bbox, frames={})
        for track_id, bbox in box.items()
    }


def propagate_tracks(
    video_id: str,
    tracks: dict[str, Track],
    predictor: VideoPredictor,
    *,
    project: Project,
) -> TrackMetadata:

    frame_ids = tuple(project.media.frame_ids(video_id))

    track_metadata = TrackMetadata(video_id, tracks, len(frame_ids))

    processed: set[int] = set()

    for batch in frame_batches(frame_ids):
        with sam2_frame_sequence(project, video_id, batch) as sequence:
            state = predictor.init_state(str(sequence.path))

            first_frame = batch[0]

            for track_id, track in tracks.items():
                if first_frame == track.initial_frame:
                    bbox = track.initial_bbox
                else:
                    previous = track.frames.get(str(first_frame))

                    if previous is None:
                        previous = track.frames.get(
                            str(frame_ids[frame_ids.index(first_frame) - 1])
                        )

                    if previous is None:
                        bbox = track.initial_bbox
                    else:
                        bbox = previous.bbox

                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(track_id),
                    box=np.asarray(bbox, dtype=np.float32),
                )

            for prediction in predictor.propagate_in_video(state):
                frame_idx = sequence.original(prediction.frame_idx)

                if frame_idx in processed:
                    continue

                processed.add(frame_idx)

                process_frame(
                    video_id,
                    prediction,
                    track_metadata,
                    project=project,
                    original_frame=frame_idx,
                )

            del state

        print(f"{video_id}: tracked {len(processed)}/{len(frame_ids)} frames.", flush=True)

    if processed != set(frame_ids):
        raise RuntimeError("SAM2 did not process the full sequence")

    return track_metadata


def process_frame(
    video_id: str,
    prediction: Prediction,
    track_metadata: TrackMetadata,
    *,
    project: Project,
    original_frame: int | None = None,
) -> None:
    frame_idx = prediction.frame_idx if original_frame is None else original_frame

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


if __name__ == "__main__":
    run(Project.open())
