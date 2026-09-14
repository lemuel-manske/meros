from src.artifacts import get_vids
from src.crop import persist_crops
from src.frame import get_frame
from src.metadata import get_track_metadata


_CROP_MARGIN = 15


def extract_n_persist_crops(vid_id: str) -> None:
    metadata = get_track_metadata(vid_id)

    for track_id, track in metadata["tracks"].items():
        frames = track["frames"]

        crops = []

        for frame_num, frame_data in frames.items():
            bbox = frame_data["bbox"]

            frame = get_frame(vid_id, int(frame_num))

            if frame is None:
                print(f"Warning: Frame {frame_num} not found for video {vid_id}. Skipping.")

                continue

            x1, y1, x2, y2 = map(int, bbox)

            frame_h, frame_w = frame.shape[:2]

            x1 = max(0, x1 - _CROP_MARGIN)
            y1 = max(0, y1 - _CROP_MARGIN)

            x2 = min(frame_w, x2 + _CROP_MARGIN)
            y2 = min(frame_h, y2 + _CROP_MARGIN)

            crop = frame[y1:y2, x1:x2]

            crops.append((int(frame_num), crop))

        persist_crops(vid_id, crops, track_id)


if __name__ == "__main__":
    """
    Extracts crops from all videos in the specified paths and saves them to disk.
    """

    for vid_id in get_vids():
        extract_n_persist_crops(vid_id)

    print("Crops extracted and saved successfully.")
