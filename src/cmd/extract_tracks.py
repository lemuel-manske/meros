from src.artifacts import get_vids
from src.frame import get_frame
from src.metadata import get_vis_metadata
from src.track import persist_tracks


def extract_n_persist_tracks(vid_id: str) -> None:
    metadata = get_vis_metadata(vid_id)

    obj = metadata["objects"]["1"]
    frames = obj["frames"]

    track_frames = []

    for frame_num, frame_data in frames.items():
        bbox = frame_data["bbox"]

        track_frame = get_frame(vid_id, int(frame_num))

        if track_frame is None:
            print(f"Warning: Frame {frame_num} not found for video {vid_id}. Skipping.")

            continue

        x1, y1, x2, y2 = map(int, bbox)

        margin = 15

        frame_h, frame_w = track_frame.shape[:2]

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)

        x2 = min(frame_w, x2 + margin)
        y2 = min(frame_h, y2 + margin)

        cropped_frame = track_frame[y1:y2, x1:x2]

        track_frames.append((int(frame_num), cropped_frame))

    persist_tracks(vid_id, track_frames)


def extract_n_persist_tracks_for_all_vids() -> None:
    for vid_id in get_vids():
        extract_n_persist_tracks(vid_id)


if __name__ == "__main__":
    """
    Extracts tracks from all videos in the specified paths and saves them to disk.
    """

    extract_n_persist_tracks_for_all_vids()

    print("Tracks extracted and saved successfully.")
