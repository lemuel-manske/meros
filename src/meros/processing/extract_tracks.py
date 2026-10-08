from __future__ import annotations

import cv2 as cv
import numpy as np

import os
import tempfile
from collections.abc import Callable
from meros.project import Project, default_project

from meros.domain.prediction import Prediction, VideoPredictor
from meros.external import build_sam2_predictor

from pathlib import Path


from meros import Track, TrackMetadata, TrackObservation


AUTO = True  # controls whether to automatically select subjects or prompt for manual selection

_QUIT_KEY = "q"
_ADD_TRACK_KEY = "a"


def run(
    project: Project = default_project,
    *,
    predictor_factory: Callable[[], VideoPredictor] | None = None,
) -> None:
    predictor = (
        predictor_factory() if predictor_factory else build_sam2_predictor(project.checkpoint)
    )

    for video in project.metadata.read_videos():
        video_id = video.video_id

        tracks = (
            manually_select_track_seeds(video_id, project=project)
            if not AUTO
            else select_known_track_seeds(video_id, project=project)
        )

        track_metadata = propagate_tracks(video_id, tracks, predictor, project=project)

        if not track_metadata.tracks or any(not t.frames for t in track_metadata.tracks.values()):
            raise RuntimeError(f"{video_id}: incomplete tracking; saved metadata was not replaced")

        project.metadata.write_track(video_id, track_metadata)

        print(f"Saved {len(tracks)} tracks for {video_id}.")


def tracks_complete(project: Project = default_project) -> bool:
    try:
        videos = project.metadata.read_videos()

        seeds = project.metadata.read_bboxes() if AUTO else {}
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

        if AUTO and set(saved.tracks) != set(video_seeds):
            return False

        for track_id, track in saved.tracks.items():
            if track.initial_frame not in frame_ids or not track.frames:
                return False

            if AUTO and (track.initial_frame != 0 or track.initial_bbox != video_seeds[track_id]):
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


def get_frame_ids(video_id: str, *, project: Project = default_project) -> list[int]:
    return project.media.frame_ids(video_id)


def select_known_track_seeds(
    video_id: str, *, project: Project = default_project
) -> dict[str, Track]:
    """
    Automatically selects the subjects to track in the video.
    Based on a previos initial bbox definition for each video,
    it will return a dictionary of tracks with the initial frame and bbox for each subject.
    """

    bboxes = project.metadata.read_bboxes()

    if video_id not in bboxes.keys():
        raise ValueError(
            f"{video_id}: no known initial bboxes available. ",
            "Please use `manually_select_track_seeds` instead.",
        )

    frame_ids = get_frame_ids(video_id, project=project)

    if not frame_ids:
        raise ValueError(f"{video_id}: no extracted frames available.")

    initial_frame = frame_ids[0]

    tracks: dict[str, Track] = {}

    for track_id, bbox in bboxes[video_id].items():
        tracks[track_id] = Track(
            initial_frame=initial_frame,
            initial_bbox=bbox,
            frames={},
        )

    return tracks


def manually_select_track_seeds(
    video_id: str, *, project: Project = default_project
) -> dict[str, Track]:
    """
    Prompts for user interaction for select the subjects to track in the video.
    """

    def draw_initial_tracks(
        frame: np.ndarray,
        frame_idx: int,
        tracks: dict[str, Track],
    ) -> np.ndarray:
        preview = frame.copy()

        for track_id, track in tracks.items():
            if track.initial_frame != frame_idx:
                continue

            x1, y1, x2, y2 = track.initial_bbox

            cv.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv.putText(
                preview,
                track_id,
                (x1, max(20, y1)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return preview

    def get_selected_frame(
        window: str,
        frame_ids: list[int],
    ) -> int:
        position = cv.getTrackbarPos(
            "Frame",
            window,
        )

        position = min(
            position,
            len(frame_ids) - 1,
        )

        return frame_ids[position]

    def add_track(
        frame: np.ndarray,
        frame_idx: int,
        tracks: dict[str, Track],
    ) -> None:
        roi_win_name = "Select a new fish"

        x, y, w, h = cv.selectROI(
            roi_win_name,
            frame,
            showCrosshair=True,
            fromCenter=False,
        )

        cv.destroyWindow(roi_win_name)

        if w <= 0 or h <= 0:
            return

        track_id = str(len(tracks) + 1)

        tracks[track_id] = Track(
            initial_frame=frame_idx,
            initial_bbox=[
                int(x),
                int(y),
                int(x + w),
                int(y + h),
            ],
            frames={},
        )

    frame_ids = get_frame_ids(video_id, project=project)

    if not frame_ids:
        raise ValueError(f"{video_id}: no extracted frames available.")

    tracks: dict[str, Track] = {}

    win_name = "Meros: a = add, q = finish"

    cv.namedWindow(
        win_name,
        cv.WINDOW_NORMAL,
    )

    cv.createTrackbar(
        "Frame",
        win_name,
        0,
        max(1, len(frame_ids) - 1),
        lambda _: None,
    )

    try:
        while (
            cv.getWindowProperty(
                win_name,
                cv.WND_PROP_VISIBLE,
            )
            >= 1
        ):
            frame_idx = get_selected_frame(
                win_name,
                frame_ids,
            )

            frame = project.media.read_frame(
                video_id,
                frame_idx,
            )

            cv.imshow(
                win_name,
                draw_initial_tracks(
                    frame,
                    frame_idx,
                    tracks,
                ),
            )

            key = cv.waitKey(30) & 0xFF

            if key == ord(_ADD_TRACK_KEY):
                add_track(
                    frame,
                    frame_idx,
                    tracks,
                )

            elif key == ord(_QUIT_KEY):
                break

    finally:
        cv.destroyAllWindows()

    if not tracks:
        raise ValueError(f"{video_id}: no fish selected.")

    return tracks


def propagate_tracks(
    video_id: str,
    tracks: dict[str, Track],
    predictor: VideoPredictor,
    *,
    project: Project = default_project,
) -> TrackMetadata:

    def add_track_prompts(
        state: object,
        tracks: dict[str, Track],
        predictor: VideoPredictor,
    ) -> None:
        for track_id, track in tracks.items():
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=track.initial_frame,
                obj_id=int(track_id),
                box=np.asarray(
                    track.initial_bbox,
                    dtype=np.float32,
                ),
            )

    track_metadata = TrackMetadata(
        video_id=video_id,
        tracks=tracks,
    )

    # SAM2 must see the full ordered sequence on every run. Diagnostic
    # images are not checkpoints and must never remove frames from inference.
    needed_frames = get_frame_ids(video_id, project=project)

    if needed_frames != list(range(len(needed_frames))):
        raise ValueError(f"{video_id}: expected contiguous zero-based frames")

    if not needed_frames:
        raise ValueError(f"{video_id}: no frames available")

    with tempfile.TemporaryDirectory(dir=project.media.frames_path(video_id).parent) as tmp_dir:
        tmp_path = Path(tmp_dir)

        frame_map: list[int] = []

        for temp_idx, original_idx in enumerate(needed_frames):
            src = project.media.frame_path(video_id, original_idx).resolve()

            dst = tmp_path / f"{temp_idx:06d}.jpg"

            os.link(src, dst)

            frame_map.append(original_idx)

        state = predictor.init_state(
            video_path=str(tmp_path),
        )

        add_track_prompts(
            state,
            tracks,
            predictor,
        )

        processed = set()

        for prediction in predictor.propagate_in_video(state):
            tmp_idx = prediction.frame_idx

            if tmp_idx in processed or not 0 <= tmp_idx < len(frame_map):
                raise RuntimeError("Invalid predictor frame index")

            processed.add(tmp_idx)

            process_frame(
                video_id,
                Prediction(frame_map[tmp_idx], prediction.masks),
                track_metadata,
                project=project,
            )

        if processed != set(range(len(frame_map))):
            raise RuntimeError("SAM2 did not process the full sequence")

        return TrackMetadata(video_id, tracks, len(processed))


def process_frame(
    video_id: str,
    prediction: Prediction,
    track_metadata: TrackMetadata,
    *,
    project: Project = default_project,
) -> None:
    frame_idx = prediction.frame_idx

    def draw_mask(
        frame: np.ndarray,
        mask: np.ndarray,
        obj_id: int,
        bbox: list[int],
    ) -> None:
        x1, y1, _, _ = bbox

        color = np.array(
            [
                (obj_id * 67) % 256,
                (obj_id * 131) % 256,
                255,
            ],
            dtype=np.uint8,
        )

        frame[mask] = (
            frame[mask].astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5
        ).astype(np.uint8)

        cv.putText(
            frame,
            str(obj_id),
            (x1, max(20, y1)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    def bbox_from_mask(
        mask: np.ndarray,
    ) -> list[int] | None:
        ys, xs = np.where(mask)

        if len(xs) == 0:
            return None

        return [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

    frame = project.media.read_frame(
        video_id,
        frame_idx,
    )

    visualization = frame.copy() if project.options.diagnostics else None

    for track_id, mask in prediction.masks.items():
        track = track_metadata.tracks[track_id]

        if frame_idx < track.initial_frame:
            continue

        if mask.shape != frame.shape[:2]:
            raise ValueError(f"{video_id}/{frame_idx}: mask and frame dimensions differ")

        bbox = bbox_from_mask(mask)

        if bbox is None:
            continue

        # Track is frozen, but the frames dict itself is still mutable.
        track.frames[str(frame_idx)] = TrackObservation(
            bbox=bbox,
            mask_area=int(mask.sum()),
        )

        if visualization is not None:
            draw_mask(
                visualization,
                mask,
                obj_id=int(track_id),
                bbox=bbox,
            )

    if visualization is not None:
        project.media.write_visualization(video_id, frame_idx, visualization)
