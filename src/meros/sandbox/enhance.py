import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import distance_transform_edt

from meros.sandbox.sources import sources


SCALES = (0.6, 0.7, 0.8, 0.9)

MANTIUK_GAMMA = 1.0
MANTIUK_SATURATION = 1.0

CLAHE_CLIP_LIMIT = 4.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def mantiuk(image: np.ndarray, scale: float) -> np.ndarray:
    valid = image[:, :, 3] > 0

    if not valid.any():
        return image.copy()

    indices = distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )

    filled = image[:, :, :3][indices]

    rgb = cv.cvtColor(filled, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)

    if np.ptp(cv.cvtColor(linear, cv.COLOR_RGB2GRAY)) == 0:
        return image.copy()

    black = np.all(linear == linear.min(), axis=2)

    mapper = cv.createTonemapMantiuk(
        gamma=MANTIUK_GAMMA,
        scale=scale,
        saturation=MANTIUK_SATURATION,
    )

    mapped = mapper.process(linear)
    mapped[black] = 0

    if not np.isfinite(mapped).all():
        raise ValueError("Mantiuk produced non-finite pixels")

    rgb8 = np.rint(np.clip(mapped, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)

    result = image.copy()
    result[:, :, :3] = cv.cvtColor(rgb8, cv.COLOR_RGB2BGR)
    result[~valid, :3] = 0

    return result


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
    for s in sources:
        image = cv.imread(s.path, cv.IMREAD_UNCHANGED)

        assert image is not None
        assert image.ndim == 3 and image.shape[2] == 4

        images = [
            ("Original", image),
            ("CLAHE", clahe(image)),
        ]

        for scale in SCALES:
            images.append((f"Mantiuk (scale={scale})", mantiuk(image, scale)))

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        for ax, (title, processed) in zip(axes.flat, images):
            show(ax, processed, title)

        fig.suptitle("Mantiuk tone mapping comparison")
        plt.tight_layout()
        plt.show()
