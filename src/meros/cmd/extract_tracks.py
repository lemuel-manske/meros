import cv2 as cv
import numpy as np
import torch

from src.meros import media, metadata, Track, TrackMetadata, TrackObservation
from src.meros.external import build_sam2_predictor, SAM2Predictor


_QUIT_KEY = "q"
_ADD_TRACK_KEY = "a"


def run() -> None:
    predictor = build_sam2_predictor()

    for video in metadata.read_videos():
        vid_id = video.video_id

        tracks = select_subjects(vid_id)
        track_metadata = propagate_tracks(
            vid_id,
            tracks,
            predictor,
        )

        metadata.write_track(
            vid_id,
            track_metadata,
        )

        print(f"Saved {len(tracks)} tracks for {vid_id}.")


def get_frame_ids(
    vid_id: str,
) -> list[int]:
    return [
        frame.frame_idx
        for frame in media.read_frames(vid_id)
    ]


def select_subjects(
    vid_id: str,
) -> dict[str, Track]:

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

    frame_ids = get_frame_ids(vid_id)

    if not frame_ids:
        raise ValueError(
            f"{vid_id}: no extracted frames available."
        )

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
                vid_id,
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
        raise ValueError(
            f"{vid_id}: no fish selected."
        )

    return tracks


def propagate_tracks(
    vid_id: str,
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
        video_id=vid_id,
        tracks=tracks,
    )

    state = predictor.init_state(
        video_path=str(media.frames_path(vid_id)),
    )

    add_track_prompts(
        state,
        tracks,
        predictor,
    )

    for (
        frame_idx,
        obj_ids,
        mask_logits,
    ) in predictor.propagate_in_video(state):
        process_frame(
            vid_id,
            frame_idx,
            obj_ids,
            mask_logits,
            track_metadata,
        )

    return track_metadata


def process_frame(
    vid_id: str,
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
            frame[mask].astype(np.float32) * 0.5
            + color.astype(np.float32) * 0.5
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
        vid_id,
        frame_idx,
    )

    visualization = frame.copy()

    for obj_id, logits in zip(
        obj_ids,
        mask_logits,
    ):
        track_id = str(int(obj_id))
        track = track_metadata.tracks[track_id]

        if frame_idx < track.initial_frame:
            continue

        mask = np.squeeze(
            (logits > 0).cpu().numpy()
        )

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

        draw_mask(
            visualization,
            mask,
            obj_id=int(obj_id),
            bbox=bbox,
        )

    media.write_visualization(
        vid_id,
        frame_idx,
        visualization,
    )
