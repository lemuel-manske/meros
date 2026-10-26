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

        requested = []

        if intervals:
            for interval in intervals:
                requested.extend(range(interval.start, interval.end + 1))
        else:
            requested = list(range(capture_len))

        if actual != requested:
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

        existing_frame_ids = {
            f.stem for f in project.media.frames_path(video.video_id).glob("*.jpg")
        }

        frame_idx = 0
        extracted_count = 0

        try:
            intervals = video.intervals

            if intervals:
                frames_to_extract = set()

                for interval in intervals:
                    frames_to_extract.update(range(interval.start, interval.end + 1))
            else:
                frames_to_extract = set(range(int(capture.get(cv.CAP_PROP_FRAME_COUNT))))

            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                if frame_idx in frames_to_extract and str(frame_idx) not in existing_frame_ids:
                    if video.mirrored:
                        frame = cv.flip(frame, 1)

                    project.media.write_frame(
                        video.video_id,
                        frame_idx,
                        frame,
                    )

                    extracted_count += 1

                frame_idx += 1

        finally:
            capture.release()

        print(
            f"{video.video_id}: "
            f"{extracted_count} new frames extracted, "
            f"{len(frames_to_extract)} requested."
        )


if __name__ == "__main__":
    run(Project.open())
