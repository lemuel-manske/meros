import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


SUBJECT = "rogerio"

CROPS = {
    9: (270, 118, 962, 616),
    12: (165, 115, 1680, 1120),
}

WIDTH = 900
HEIGHT = 630

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (9, 7)


def fit_canvas(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]

    scale = min(WIDTH / width, HEIGHT / height)

    interpolation = cv.INTER_AREA if scale < 1 else cv.INTER_CUBIC

    scaled = cv.resize(
        image, (round(width * scale), round(height * scale)), interpolation=interpolation
    )

    canvas = np.full((HEIGHT, WIDTH, 3), (28, 28, 28), dtype=np.uint8)

    y = (HEIGHT - scaled.shape[0]) // 2
    x = (WIDTH - scaled.shape[1]) // 2

    canvas[y : y + scaled.shape[0], x : x + scaled.shape[1]] = scaled

    return canvas


def clahe(image: np.ndarray) -> np.ndarray:
    light, channel_a, channel_b = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))

    enhanced = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID).apply(light)

    return cv.cvtColor(cv.merge((enhanced, channel_a, channel_b)), cv.COLOR_LAB2BGR)


def run() -> None:
    project = Project.open()

    figure, axes = plt.subplots(2, len(CROPS), figsize=(15, 10))

    for column, (image_id, (x1, y1, x2, y2)) in enumerate(CROPS.items()):
        image = project.media.read_subject_image(SUBJECT, image_id)

        height, width = image.shape[:2]

        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Crop for {SUBJECT}/{image_id} exceeds source image bounds")

        original = fit_canvas(image[y1:y2, x1:x2])
        enhanced = clahe(original)

        for row, view in enumerate((original, enhanced)):
            axis = axes[row, column]
            axis.imshow(cv.cvtColor(view, cv.COLOR_BGR2RGB))
            axis.set_title(f"{SUBJECT}/{image_id} — {'Original' if row == 0 else 'CLAHE'}")
            axis.set_axis_off()

    figure.tight_layout()

    plt.show(block=True)
    plt.close(figure)


if __name__ == "__main__":
    run()
