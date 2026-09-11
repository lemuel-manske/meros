import cv2 as cv

VIDEO_FOLDER = "/home/lkmliz/dev/meros/videos"
FRAMES_FOLDER = "/home/lkmliz/dev/meros/frames"

VIDEO_PATHS = [
    VIDEO_FOLDER + "/260113_S3_SC_Monoboia_0025_1#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0025_2#_3#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0026_4#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0039_Diversos#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0065_5#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0065_6#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0065_7#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0065_8#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0066_10#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0066_9#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0101_11#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0101_12#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0102_13#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0102_14#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0102_15#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0147_16#.mov",
    # VIDEO_FOLDER + "260113_S3_SC_Monoboia_0149_Diversos#.mov",
]

def process_frames(video_path, frame_interval=2):
    """
    Process frames from a video file at a specified interval.
    """
    cap = cv.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        reached_end = not ret

        if reached_end:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    return frames


def save_frames(frames, output_folder, base_filename):
    """
    Save frames to the specified output folder with a base filename.
    """
    for idx, frame in enumerate(frames):
        filename = f"{output_folder}/{base_filename}_frame_{idx}.jpg"
        cv.imwrite(filename, frame)
        print(f"Saved frame {idx} to {filename}")


def get_vid_id(video_path):
    """
    Extract the video ID from the video path.
    """
    return video_path.split("/")[-1].split(".")[0]


if __name__ == "__main__":
    for video_path in VIDEO_PATHS:
        frames = process_frames(video_path, frame_interval=2)
        base_filename = get_vid_id(video_path)
        save_frames(frames, FRAMES_FOLDER, base_filename)
