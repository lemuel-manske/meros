import math

import numpy as np

from meros.domain.storage import BBoxPrediction, Video
from meros.models.yolo import YOLO
from meros.project import Project
from meros.utils import configure_torch

FRAME_COUNT = 300
SCAN_MULTIPLIER = 2
IMAGE_SIZE = 640
CONFIDENCE = 0.25


def allocate_frames(videos: list[Video]) -> list[int]:
    capacities = [len(video.frames) for video in videos]

    total = min(FRAME_COUNT, sum(capacities))

    weights = np.sqrt([len(video.frames) for video in videos])
    desired = total * weights / weights.sum()

    counts = [min(capacity, int(count)) for capacity, count in zip(capacities, desired)]

    while sum(counts) < total:
        index = max(
            (i for i in range(len(videos)) if counts[i] < capacities[i]),
            key=lambda i: desired[i] - counts[i],
        )

        counts[index] += 1

    return counts


def prepare_video(project: Project, model, video: Video, count: int) -> None:
    if count == 0:
        return

    existing = sum(frame.selected for frame in video.frames)

    if existing >= count:
        print(f"{video._id}: {existing} frames already prepared", flush=True)

        return

    pool_size = min(SCAN_MULTIPLIER * count, len(video.frames))

    indexes = np.linspace(0, len(video.frames) - 1, pool_size, dtype=int)
    uniform = set(np.linspace(0, pool_size - 1, math.ceil(count / 2), dtype=int).tolist())

    ranked: list[tuple[float, int, list[BBoxPrediction]]] = []

    for step, index in enumerate(indexes):
        frame = video.frames[int(index)]

        if frame.selected:
            continue

        image = frame.im

        result = model.predict(
            image, imgsz=IMAGE_SIZE, conf=CONFIDENCE, device="cpu", verbose=False
        )[0]

        height, width = image.shape[:2]

        suggestions = []
        priority = 0.0

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox = [
                max(0, math.floor(x1)),
                max(0, math.floor(y1)),
                min(width, math.ceil(x2)),
                min(height, math.ceil(y2)),
            ]

            suggestions.append(BBoxPrediction(bbox=bbox, confidence=float(box.conf.item())))

            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)
            priority = max(priority, area * float(box.conf.item()))

        if step in uniform and existing < count:
            frame.bboxes = suggestions
            frame.selected = True
            existing += 1
        else:
            ranked.append((priority, int(index), suggestions))

        if step % 10 == 0:
            print(f"{video._id}: scanned {step + 1}/{pool_size}", flush=True)

    for _, index, suggestions in sorted(ranked, reverse=True):
        if existing >= count:
            break

        frame = video.frames[index]
        frame.bboxes = suggestions
        frame.selected = True
        existing += 1

    project.videos.write(video)

    print(f"{video._id}: prepared {existing} frames for review", flush=True)


def run(project: Project) -> None:
    configure_torch()

    videos = [video for video in project.videos.get_all() if video.frames]

    assert videos, "No videos available"

    model = YOLO(project.fishial_model)

    for video, count in zip(videos, allocate_frames(videos), strict=True):
        prepare_video(project, model, video, count)


if __name__ == "__main__":
    run(Project.open())
