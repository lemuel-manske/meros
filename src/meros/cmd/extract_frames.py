import cv2 as cv

from src.meros import media, metadata


def run() -> None:
    for video in metadata.read_videos():
        capture = cv.VideoCapture(video.path)

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {video.path}"
            )

        frame_idx = 0

        try:
            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                media.write_frame(
                    video.video_id,
                    frame_idx,
                    frame,
                )

                frame_idx += 1

        finally:
            capture.release()

        if frame_idx == 0:
            raise RuntimeError(
                f"{video.video_id}: no frames extracted."
            )

        print(
            f"{video.video_id}: extracted {frame_idx} frames."
        )
