import cv2 as cv

from meros.project import Project, default_project


def frames_complete(project: Project = default_project) -> bool:
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

            expected = video.frame_count or int(capture.get(cv.CAP_PROP_FRAME_COUNT))

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


def run(project: Project = default_project) -> None:
    for video in project.metadata.read_videos():
        capture = cv.VideoCapture(video.path)

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video.path}")

        # Remove obsolete frames from an earlier extraction only after opening.
        for path in project.media.frames_path(video.video_id).glob("*.jpg"):
            path.unlink()

        frame_idx = 0

        try:
            while True:
                ok, frame = capture.read()

                if not ok:
                    break

                if video.mirrored:
                    frame = cv.flip(frame, 1)

                project.media.write_frame(
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
