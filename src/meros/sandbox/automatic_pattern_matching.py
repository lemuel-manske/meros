"""
Explore automatic correspondences between head-pattern textures, without manual regions.

Run from the repository root:
    python -m meros.sandbox.automatic_pattern_matching

This is a diagnostic of visual correspondence, not individual identification.
All photos are searched without human landmarks or spot coordinates.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


PAIRS = (
    (("rogerio", 9), ("rogerio", 12)),
    (("rafael", 3), ("rogerio", 9)),
)

MAX_WIDTH = 680
WINDOW_SIZES = (64, 96)
MAX_WINDOWS = 90
DESCRIPTOR_SIZE = 24
ROTATIONS = (-15, 0, 15)
SCALES = (0.9, 1.0, 1.1)
MIN_SPATIAL_INLIERS = 4
RANSAC_THRESHOLD = 28.0
MAX_DRAWN = 24


def prepare(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Create image-space normalized fine/coarse tonal responses.
    """
    height, width = image.shape[:2]
    factor = min(1.0, MAX_WIDTH / width)
    resized = cv.resize(image, (round(width * factor), round(height * factor)))

    light = cv.cvtColor(resized, cv.COLOR_BGR2LAB)[:, :, 0]
    light = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    values = light.astype(np.float32)

    local = cv.GaussianBlur(values, (0, 0), sigmaX=3)
    medium = cv.GaussianBlur(values, (0, 0), sigmaX=11)
    broad = cv.GaussianBlur(values, (0, 0), sigmaX=29)

    return resized, (local - medium, medium - broad)


def candidate_windows(
    maps: tuple[np.ndarray, np.ndarray],
) -> list[tuple[int, int, int]]:
    """
    Propose high-contrast overlapping windows independently in each photograph.
    """
    fine, medium = maps
    height, width = fine.shape
    proposals: list[tuple[float, int, int, int]] = []

    for size in WINDOW_SIZES:
        stride = size // 2

        if height < size or width < size:
            continue

        for y in range(0, height - size + 1, stride):
            for x in range(0, width - size + 1, stride):
                patch_fine = fine[y : y + size, x : x + size]
                patch_medium = medium[y : y + size, x : x + size]
                energy = float(np.std(patch_fine) + np.std(patch_medium))
                proposals.append((energy, x, y, size))

    proposals.sort(reverse=True)
    selected: list[tuple[int, int, int]] = []

    for _, x, y, size in proposals:
        cx, cy = x + size / 2, y + size / 2

        if any(
            (cx - (px + ps / 2)) ** 2 + (cy - (py + ps / 2)) ** 2 < 26**2
            for px, py, ps in selected
        ):
            continue

        selected.append((x, y, size))

        if len(selected) == MAX_WINDOWS:
            break

    return selected


def descriptor(
    maps: tuple[np.ndarray, np.ndarray],
    window: tuple[int, int, int],
    rotation: float = 0,
    scale: float = 1,
) -> np.ndarray:
    x, y, size = window
    channels = []

    for response in maps:
        patch = response[y : y + size, x : x + size]
        patch = cv.resize(patch, (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE))

        if rotation != 0 or scale != 1:
            center = ((DESCRIPTOR_SIZE - 1) / 2, (DESCRIPTOR_SIZE - 1) / 2)
            matrix = cv.getRotationMatrix2D(center, rotation, scale)
            patch = cv.warpAffine(
                patch,
                matrix,
                (DESCRIPTOR_SIZE, DESCRIPTOR_SIZE),
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_REFLECT_101,
            )

        patch = patch - patch.mean()
        patch = patch / max(float(patch.std()), 1e-6)
        channels.append(patch.ravel())

    vector = np.concatenate(channels)
    return vector / max(float(np.linalg.norm(vector)), 1e-6)


def mutual_correspondences(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match local texture under a small set of in-plane viewpoints.
    """
    first_windows = candidate_windows(first)
    second_windows = candidate_windows(second)

    if not first_windows or not second_windows:
        return np.empty((0, 2), dtype=int), np.empty(0), np.empty((0, 2)), np.empty((0, 2))

    first_features = np.stack([descriptor(first, window) for window in first_windows])
    transformed = np.stack(
        [
            descriptor(second, window, angle, scale)
            for window in second_windows
            for angle in ROTATIONS
            for scale in SCALES
        ]
    )

    similarity = first_features @ transformed.T
    count_variants = len(ROTATIONS) * len(SCALES)
    similarity = similarity.reshape(len(first_windows), len(second_windows), count_variants)
    best = similarity.max(axis=2)

    forward = best.argmax(axis=1)
    reverse = best.argmax(axis=0)
    matched = np.flatnonzero(reverse[forward] == np.arange(len(forward)))
    targets = forward[matched]
    scores = best[matched, targets]
    order = np.argsort(-scores)

    pairs = np.column_stack((matched[order], targets[order]))
    centers_first = np.array([(x + s / 2, y + s / 2) for x, y, s in first_windows])
    centers_second = np.array([(x + s / 2, y + s / 2) for x, y, s in second_windows])

    return pairs, scores[order], centers_first, centers_second


def geometric_consistency(
    pairs: np.ndarray, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """
    Flag similarity-consistent matches for visualization; not identity evidence.
    """
    if len(pairs) < MIN_SPATIAL_INLIERS:
        return np.zeros(len(pairs), dtype=bool)

    _, inliers = cv.estimateAffinePartial2D(
        first[pairs[:, 0]].astype(np.float32),
        second[pairs[:, 1]].astype(np.float32),
        method=cv.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=2000,
    )

    if inliers is None:
        return np.zeros(len(pairs), dtype=bool)

    return inliers.ravel().astype(bool)


def visualize(
    first_image: np.ndarray,
    second_image: np.ndarray,
    pairs: np.ndarray,
    first_centers: np.ndarray,
    second_centers: np.ndarray,
    inliers: np.ndarray,
    name: str,
) -> None:
    height = max(first_image.shape[0], second_image.shape[0])
    width = first_image.shape[1]
    canvas = np.zeros((height, width + second_image.shape[1], 3), dtype=np.uint8)
    canvas[: first_image.shape[0], :width] = cv.cvtColor(first_image, cv.COLOR_BGR2RGB)
    canvas[: second_image.shape[0], width:] = cv.cvtColor(second_image, cv.COLOR_BGR2RGB)

    figure, axis = plt.subplots(figsize=(17, 9))
    axis.imshow(canvas)

    for (i, j), inlier in zip(pairs[:MAX_DRAWN], inliers[:MAX_DRAWN], strict=True):
        start = first_centers[i]
        end = second_centers[j] + np.array([width, 0])
        color = "#2bd889" if inlier else "#e29a4b"
        axis.plot((start[0], end[0]), (start[1], end[1]), color=color, alpha=0.7)
        axis.scatter((start[0], end[0]), (start[1], end[1]), s=13, color=color)

    axis.set_title(f"{name}: {len(pairs)} mutual proposals, {inliers.sum()} similarity inliers")
    axis.set_axis_off()
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


def run() -> None:
    project = Project.open()

    for first_id, second_id in PAIRS:
        first, first_maps = prepare(project.media.read_subject_image(*first_id))
        second, second_maps = prepare(project.media.read_subject_image(*second_id))
        pairs, scores, first_centers, second_centers = mutual_correspondences(
            first_maps, second_maps
        )
        inliers = geometric_consistency(pairs, first_centers, second_centers)
        name = f"{first_id} vs {second_id}"

        highest = f"{scores[0]:.3f}" if len(scores) else "n/a"
        print(
            f"{name}: {len(pairs)} mutual proposals, {int(inliers.sum())} "
            f"similarity inliers, highest cosine={highest} (not identity evidence)"
        )

        visualize(first, second, pairs, first_centers, second_centers, inliers, name)


if __name__ == "__main__":
    run()
