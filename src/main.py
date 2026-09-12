from . import get_vid_id, get_annotations, path_to_track, path_to_vid, VIDEOS
from .cvat_processor import parse

import cv2 as cv


def crop_subject_from_vid(vid_path: str) -> list[cv.Mat]:
    """
    Crop the subject from a video based on annotations from a CVAT XML file.
    """
    vid_id = get_vid_id(vid_path)
    track_path = get_annotations(vid_id)
    annotations = parse(track_path)

    cap = cv.VideoCapture(vid_path)

    cropped_frames = []

    for annotation in annotations:
        frame_number = annotation['frame_number']
        xtl = int(annotation['xtl'])
        ytl = int(annotation['ytl'])
        xbr = int(annotation['xbr'])
        ybr = int(annotation['ybr'])

        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {frame_number} from video.")

            continue

        cropped_frame = frame[ytl:ybr, xtl:xbr]
        cropped_frames.append(cropped_frame)

    return cropped_frames


def persist_frames(frames: list[cv.Mat], output_dir: str):
    """
    Persist frames to disk.
    """
    for i, frame in enumerate(frames):
        cv.imwrite(f"{output_dir}/frame_{i}.jpg", frame)


def process_videos():
    """
    Process all videos in the VIDEOS list.
    """
    for vid_id in VIDEOS:
        vid_path = path_to_vid(vid_id)
        cropped_frames = crop_subject_from_vid(vid_path)
        output_dir = path_to_track(vid_id)
        persist_frames(cropped_frames, output_dir)


if __name__ == "__main__":
    process_videos()
