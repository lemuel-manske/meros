"""Shared helpers for manually comparing head-spot detectors in the sandbox."""

from collections.abc import Callable

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


type Detector = Callable[[np.ndarray, np.ndarray], np.ndarray]


def iter_frontal_crops():
    project = Project.open()

    for video in project.metadata.read_videos():
        for annotation in video.matching_annotations(viewpoint="frontal", min_quality=3):
            path = project.media.masked_crop_path(
                video.video_id,
                annotation.track_id,
                (annotation.start_frame + annotation.end_frame) // 2,
            )
            image = cv.imread(str(path), cv.IMREAD_UNCHANGED)

            if image is None:
                raise FileNotFoundError(path)

            if image.ndim == 3 and image.shape[2] == 4:
                mask = (image[:, :, 3] == 255).astype(np.uint8) * 255
                bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
            else:
                mask = np.full(image.shape[:2], 255, dtype=np.uint8)
                bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR) if image.ndim == 2 else image

            mask = cv.erode(mask, np.ones((5, 5), dtype=np.uint8))
            gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

            yield video.video_id, annotation.track_id, bgr, gray, mask


def display(name: str, detector: Detector) -> None:
    for video_id, track_id, bgr, gray, mask in iter_frontal_crops():
        points = detector(gray, mask)
        annotated = bgr.copy()

        for x, y in points:
            cv.circle(annotated, (round(float(x)), round(float(y))), 4, (0, 255, 255), 1)

        figure, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        axes[0].set_title("Original")
        axes[1].imshow(cv.cvtColor(annotated, cv.COLOR_BGR2RGB))
        axes[1].set_title(f"{name}: {len(points)} centers")

        for axis in axes:
            axis.axis("off")

        figure.suptitle(f"{video_id} / {track_id}")
        figure.tight_layout()
        plt.show()
        plt.close(figure)
