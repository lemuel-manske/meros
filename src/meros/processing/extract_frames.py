import cv2 as cv

from meros.project import Project


def frames_complete(project: Project) -> bool:
    try:
        videos = project.metadata.read_videos()
    except (OSError, ValueError, KeyError, TypeError):
        return False

    if not videos:
        return False

    for video in videos:
        capture = cv.VideoCapture(video.path)

        try:
            if not capture.isOpened():
                return False

            intervals = video.intervals

            capture_len = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

            expected = 0
            if len(intervals) > 0:
                for interval in intervals:
                    if interval.start < 0 or interval.end >= capture_len:
                        return False

                    expected += interval.end - interval.start + 1
            else:
                expected = capture_len

            shape = (
                int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
                int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
            )
        finally:
            capture.release()

        if expected <= 0:
            return False

        try:
            actual = project.media.frame_ids(video.video_id)
        except ValueError:
            return False

        if actual != list(range(expected)):
            return False

        for idx in actual:
            if not project.media.valid_image(
                project.media.frame_path(video.video_id, idx), 3, shape
            ):
                return False

    return True


def run(project: Project) -> None:
    for video in project.metadata.read_videos():
        capture = cv.VideoCapture(video.path)

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video.path}")

        existing_frames_ids = [
            f.stem for f in project.media.frames_path(video.video_id).glob("*.jpg")
        ]

        frame_idx = 0

        try:
            frames_to_extract = []

            intervals = video.intervals
            if len(intervals) > 0:
                for interval in intervals:
                    frames_to_extract.extend(range(interval.start, interval.end + 1))
            else:
                frames_to_extract = list(range(int(capture.get(cv.CAP_PROP_FRAME_COUNT))))

            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                if video.mirrored:
                    frame = cv.flip(frame, 1)

                # video sources never change, so we can skip frames that already exist
                if frame_idx in frames_to_extract and str(frame_idx) not in existing_frames_ids:
                    project.media.write_frame(
                        video.video_id,
                        frame_idx,
                        frame,
                    )

                frame_idx += 1

        finally:
            capture.release()

        count = len(frames_to_extract)

        if count == 0:
            raise RuntimeError(f"{video.video_id}: no frames extracted.")

        print(f"{video.video_id}: extracted {count} frames.")
