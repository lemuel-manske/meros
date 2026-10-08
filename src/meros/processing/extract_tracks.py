from __future__ import annotations

import cv2 as cv
import numpy as np

import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from meros.external import SAM2Predictor

from pathlib import Path

from meros.config import options

from meros import media, metadata, Track, TrackMetadata, TrackObservation


AUTO = True  # controls whether to automatically select subjects or prompt for manual selection

_QUIT_KEY = "q"
_ADD_TRACK_KEY = "a"


def run() -> None:
    import torch
    from meros.external import build_sam2_predictor

    predictor = build_sam2_predictor()

    for video in metadata.read_videos():
        video_id = video.video_id

        tracks = (
            manually_select_track_seeds(video_id)
            if not AUTO
            else select_known_track_seeds(video_id)
        )

        with torch.inference_mode():
            track_metadata = propagate_tracks(video_id, tracks, predictor)

        if not track_metadata.tracks or any(not t.frames for t in track_metadata.tracks.values()):
            raise RuntimeError(f"{video_id}: incomplete tracking; saved metadata was not replaced")

        metadata.write_track(
            video_id,
            track_metadata,
        )

        print(f"Saved {len(tracks)} tracks for {video_id}.")


def tracks_complete() -> bool:
    videos = metadata.read_videos()

    if not videos:
        return False

    for video in videos:
        try:
            saved = metadata.read_track(video.video_id)
        except (FileNotFoundError, ValueError, KeyError):
            return False

        frame_ids = set(get_frame_ids(video.video_id))

        if not frame_ids or not saved.tracks or saved.processed_frame_count != len(frame_ids):
            return False

        seeds = metadata.read_bboxes().get(video.video_id, {})

        if AUTO and set(saved.tracks) != set(seeds):
            return False

        for track in saved.tracks.values():
            observed = {int(idx) for idx in track.frames}

            if not observed or not observed <= frame_ids:
                return False

    return True


def get_frame_ids(
    video_id: str,
) -> list[int]:
    return media.frame_ids(video_id)


def select_known_track_seeds(
    video_id: str,
) -> dict[str, Track]:
    """
    Automatically selects the subjects to track in the video.
    Based on a previos initial bbox definition for each video,
    it will return a dictionary of tracks with the initial frame and bbox for each subject.
    """

    bboxes = metadata.read_bboxes()

    if video_id not in bboxes.keys():
        raise ValueError(
            f"{video_id}: no known initial bboxes available. ",
            "Please use `manually_select_track_seeds` instead.",
        )

    frame_ids = get_frame_ids(video_id)

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
    video_id: str,
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

    frame_ids = get_frame_ids(video_id)

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

            frame = media.read_frame(
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
    predictor: SAM2Predictor,
) -> TrackMetadata:

    def add_track_prompts(
        state: dict[str, torch.Tensor],
        tracks: dict[str, Track],
        predictor: SAM2Predictor,
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
    needed_frames = get_frame_ids(video_id)

    if needed_frames != list(range(len(needed_frames))):
        raise ValueError(f"{video_id}: expected contiguous zero-based frames")

    if not needed_frames:
        raise ValueError(f"{video_id}: no frames available")

    with tempfile.TemporaryDirectory(dir=media.frames_path(video_id).parent) as tmp_dir:
        tmp_path = Path(tmp_dir)

        frame_map: list[int] = []

        for temp_idx, original_idx in enumerate(needed_frames):
            src = media.frame_path(video_id, original_idx).resolve()

            dst = tmp_path / f"{temp_idx:06d}.jpg"

            os.link(src, dst)

            frame_map.append(original_idx)

        # If there are no frames to process, return the track metadata without running the predictor.
        if not frame_map:
            return track_metadata

        state = predictor.init_state(
            video_path=str(tmp_path),
        )

        add_track_prompts(
            state,
            tracks,
            predictor,
        )

        processed = set()

        for (
            tmp_idx,
            obj_ids,
            mask_logits,
        ) in predictor.propagate_in_video(state):
            if tmp_idx in processed or not 0 <= tmp_idx < len(frame_map):
                raise RuntimeError("Invalid SAM2 frame index")

            processed.add(tmp_idx)

            frame_idx = frame_map[tmp_idx]

            process_frame(
                video_id,
                frame_idx,
                obj_ids,
                mask_logits,
                track_metadata,
            )

        if processed != set(range(len(frame_map))):
            raise RuntimeError("SAM2 did not process the full sequence")

        return TrackMetadata(video_id, tracks, len(processed))


def process_frame(
    video_id: str,
    frame_idx: int,
    obj_ids: torch.Tensor,
    mask_logits: torch.Tensor,
    track_metadata: TrackMetadata,
) -> None:

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

    frame = media.read_frame(
        video_id,
        frame_idx,
    )

    visualization = frame.copy() if options.diagnostics else None

    for obj_id, logits in zip(
        obj_ids,
        mask_logits,
    ):
        track_id = str(int(obj_id))

        track = track_metadata.tracks[track_id]

        if frame_idx < track.initial_frame:
            continue

        mask = np.squeeze((logits > 0).cpu().numpy())

        if mask.shape != frame.shape[:2]:
            print(
                f"Warning: mask shape {mask.shape} does not match "
                f"frame shape {frame.shape[:2]} for frame {frame_idx}."
            )

            continue

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
                obj_id=int(obj_id),
                bbox=bbox,
            )

    if visualization is not None:
        media.write_visualization(video_id, frame_idx, visualization)
