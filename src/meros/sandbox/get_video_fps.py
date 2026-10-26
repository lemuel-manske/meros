import argparse
import cv2 as cv

# returns the fps of the video at the given path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the FPS of a video")

    parser.add_argument("path", help="Path to the video file")

    args = parser.parse_args()

    path = args.path

    cap = cv.VideoCapture(path)
    assert cap.isOpened()

    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"FPS: {fps}")
