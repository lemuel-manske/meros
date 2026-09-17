from src.meros import media, metadata


CROP_MARGIN = 15


def run():
    for video in metadata.read_videos():
        for track_id, track in metadata.read_track(video.video_id).tracks.items():
            frames = track.frames

            for frame_num, frame_data in frames.items():
                bbox = frame_data.bbox

                frame = media.read_frame(video.video_id, int(frame_num))

                if frame is None:
                    print(f"Warning: Frame {frame_num} not found for video {video.video_id}. Skipping.")

                    continue

                x1, y1, x2, y2 = map(int, bbox)

                frame_h, frame_w = frame.shape[:2]

                x1 = max(0, x1 - CROP_MARGIN)
                y1 = max(0, y1 - CROP_MARGIN)

                x2 = min(frame_w, x2 + CROP_MARGIN)
                y2 = min(frame_h, y2 + CROP_MARGIN)

                crop = frame[y1:y2, x1:x2]

                media.write_crop(video.video_id, track_id, int(frame_num), crop)

    print("Crops extracted and saved successfully.")
