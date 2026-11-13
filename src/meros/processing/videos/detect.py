import math

import cv2 as cv
import numpy as np
from ultralytics.engine.results import Results

from meros.domain.storage import FrameDetection, FrameSelection
from meros.models.yolo import YOLO
from meros.project import Project
from meros.utils import configure_torch

VIDEO_ID = "GX010349"

IMAGE_SIZE = 640
DEVICE = 0
CONFIDENCE = 0.25
MIN_AREA_RATIO = 0.08
MIN_SHARPNESS = 30.0
SHARPNESS_SIZE = 256
WINDOW_SECONDS = 2
PROGRESS_INTERVAL = 100


def measure_sharpness(crop: np.ndarray) -> float:
    height, width = crop.shape[:2]
    scale = SHARPNESS_SIZE / max(height, width)

    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    gray = cv.resize(
        gray, (round(width * scale), round(height * scale)), interpolation=cv.INTER_AREA
    )
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    dx = cv.Sobel(gray, cv.CV_32F, 1, 0)
    dy = cv.Sobel(gray, cv.CV_32F, 0, 1)

    return float(cv.magnitude(dx, dy).mean())


def select_detections(image: np.ndarray, detections: list[FrameDetection]) -> list[FrameSelection]:
    height, width = image.shape[:2]
    selected = []

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        area_ratio = (x2 - x1) * (y2 - y1) / (width * height)

        if area_ratio < MIN_AREA_RATIO:
            continue

        sharpness = measure_sharpness(image[y1:y2, x1:x2])

        if sharpness < MIN_SHARPNESS:
            continue

        selected.append(
            FrameSelection(
                bbox=detection.bbox,
                confidence=detection.confidence,
                area_ratio=area_ratio,
                sharpness=sharpness,
            )
        )

    return selected


def run(project: Project) -> None:
    configure_torch()

    video = project.videos.get(VIDEO_ID)
    model = YOLO(project.mero_detector)

    assert model.names == {0: "mero"}

    detections: dict[int, list[FrameDetection]] = {}
    windows: dict[int, tuple[int, list[FrameSelection]]] = {}

    results = model.predict(
        source=video.path,
        stream=True,
        vid_stride=1,
        batch=1,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        save=False,
        verbose=False,
    )

    for index, result in enumerate(results):
        assert isinstance(result, Results) and result.boxes is not None

        height, width = result.orig_shape
        items = []

        for x1, y1, x2, y2, confidence, _ in result.boxes.cpu().data.tolist():
            items.append(
                FrameDetection(
                    bbox=[
                        max(0, math.floor(x1)),
                        max(0, math.floor(y1)),
                        min(width, math.ceil(x2)),
                        min(height, math.ceil(y2)),
                    ],
                    confidence=confidence,
                )
            )

        detections[index] = items

        selected = select_detections(result.orig_img, items)

        if selected:
            window = int(video.frames[index].timestamp // WINDOW_SECONDS)
            previous = windows.get(window)

            if previous is None or max(item.sharpness for item in selected) > max(
                item.sharpness for item in previous[1]
            ):
                windows[window] = (index, selected)

        if (index + 1) % PROGRESS_INTERVAL == 0:
            print(f"{video.video_id}: {index + 1}/{len(video.frames)} frames", flush=True)

    assert len(detections) == len(video.frames), "Video was not fully decoded"

    selection = {index: items for index, items in sorted(windows.values())}

    project.videos.write_video_detections(video, detections)
    project.videos.write_video_selection(video, selection)


if __name__ == "__main__":
    run(Project.open())
