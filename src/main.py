import cv2 as cv


_VIDEOS_FOLDER = "/home/lkmliz/dev/meros/videos"
_FRAMES_FOLDER = "/home/lkmliz/dev/meros/frames"


VIDEOS = [
    "260113_S3_SC_Monoboia_0025_1#.mov",
]


def path_to_vid(vid_id):
    """
    Get the path to a video given its ID.
    """
    return f"{_VIDEOS_FOLDER}/{vid_id}"


def get_vid_id(vid_path):
    """
    Extract the video ID from the video path.
    """
    return vid_path.split("/")[-1].split(".")[0]


def extract_frames(vid_path: str) -> list[cv.Mat]:
    """
    Extract frames from a video file.

    Args:
        vid_path (str): Path to the video file.
    """
    cap = cv.VideoCapture(vid_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def persist_frames(frames: list[cv.Mat], vid_id: str) -> None:
    """
    Persist frames to disk.

    Args:
        frames (list[cv.Mat]): List of frames to persist.
        vid_id (str): Video ID to use for naming the frames.
    """
    for idx, frame in enumerate(frames):
        persist_frame(frame, vid_id, idx)


def persist_frame(frame: cv.Mat, _: str, frame_idx: int) -> None:
    """
    Persist a single frame to disk.

    Args:
        frame (cv.Mat): Frame to persist.
        vid_id (str): Video ID to use for naming the frame.
        frame_idx (int): Index of the frame in the video.
    """
    cv.imwrite(f"{_FRAMES_FOLDER}/{frame_idx}.jpg", frame)


def extract_n_persist_frames_for_all_vids():
    for vid in VIDEOS:
        vid_path = path_to_vid(vid)
        vid_id = get_vid_id(vid_path)
        frames = extract_frames(vid_path)
        persist_frames(frames, vid_id)


if __name__ == "__main__":
    extract_n_persist_frames_for_all_vids()
