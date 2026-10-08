import cv2 as cv

from meros import media, metadata


def frames_complete() -> bool:
    videos = metadata.read_videos()

    if not videos:
        return False

    for video in videos:
        expected = video.frame_count

        if expected is None:
            capture = cv.VideoCapture(video.path)

            try:
                if not capture.isOpened():
                    return False

                expected = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()

        if expected <= 0:
            return False

        actual = set(media.frame_ids(video.video_id))

        if actual != set(range(expected)):
            return False

    return True


def run() -> None:
    for video in metadata.read_videos():
        capture = cv.VideoCapture(video.path)

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video.path}")

        # Remove obsolete frames from an earlier extraction only after opening.
        for path in media.frames_path(video.video_id).glob("*.jpg"):
            path.unlink()

        frame_idx = 0

        try:
            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                if video.mirrored:
                    frame = cv.flip(frame, 1)

                media.write_frame(
                    video.video_id,
                    frame_idx,
                    frame,
                )

                frame_idx += 1

        finally:
            capture.release()

        if frame_idx == 0:
            raise RuntimeError(f"{video.video_id}: no frames extracted.")

        print(f"{video.video_id}: extracted {frame_idx} frames.")
