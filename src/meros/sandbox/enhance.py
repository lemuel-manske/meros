import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


CLAHE_CLIP_LIMIT = 4.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def clahe(image: np.ndarray) -> np.ndarray:
    valid = image[:, :, 3] > 0
    gray = cv.cvtColor(image[:, :, :3], cv.COLOR_BGR2GRAY)

    if not valid.any():
        return image.copy()

    filled = gray.copy()
    filled[~valid] = int(np.median(gray[valid]))

    enhanced = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE).apply(
        filled
    )

    result = cv.cvtColor(enhanced, cv.COLOR_GRAY2BGRA)
    result[:, :, 3] = image[:, :, 3]
    result[~valid, :3] = 0

    return result


def show(ax, image: np.ndarray, title: str) -> None:
    rgba = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)

    ax.imshow(rgba)
    ax.set_title(title)
    ax.axis("off")


if __name__ == "__main__":
    project = Project.open()

    for vid in project.metadata.read_videos():
        annotations = vid.matching_annotations(
            viewpoint="frontal",
            min_quality=3,
        )

        for annotation in annotations:
            frame = project.media.masked_crop_path(
                vid.video_id,
                annotation.track_id,
                (annotation.start_frame + annotation.end_frame) // 2
            )

            image = cv.imread(frame, cv.IMREAD_UNCHANGED)

            assert image is not None
            assert image.ndim == 3 and image.shape[2] == 4

            images = [
                ("Original", image),
                ("CLAHE", clahe(image)),
            ]

            fig, axes = plt.subplots(1, 2, figsize=(15, 9))

            for ax, (title, processed) in zip(axes.flat, images):
                show(ax, processed, title)

            fig.suptitle("Mantiuk tone mapping comparison")
            plt.tight_layout()
            plt.show()
