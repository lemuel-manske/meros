from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes

from src.meros.processing.match_images import match_images, MatchStats


# map is_same_individual to a pair
paths = {
    True: (
        "data/media/masked_crops/65_7/0/210.png",
        "data/media/masked_crops/65_7/0/179.png",
    ),
    False: (
        "data/media/masked_crops/65_7/0/210.png",
        "data/media/masked_crops/101_11/0/250.png",
    ),
}


@dataclass(frozen=True)
class Pair:
    source: np.ndarray
    target: np.ndarray

    is_same_individual: bool


def read_pairs() -> list[Pair]:
    r = []

    for is_same_individual, (s, t) in paths.items():
        source_img = cv.imread(s, cv.IMREAD_UNCHANGED)
        assert source_img is not None, f"Could not load {s}"

        target_img = cv.imread(t, cv.IMREAD_UNCHANGED)
        assert target_img is not None, f"Could not load {t}"

        r.append(
            Pair(
                source=cv.cvtColor(source_img, cv.COLOR_BGR2BGRA),
                target=cv.cvtColor(target_img, cv.COLOR_BGR2BGRA),
                is_same_individual=is_same_individual,
            )
        )

    return r


def draw_matches(
    ax: Axes,
    source: np.ndarray,
    target: np.ndarray,
    result: MatchStats,
    inliers_only: bool = False,
):
    source_rgb = cv.cvtColor(source, cv.COLOR_BGRA2RGBA)
    target_rgb = cv.cvtColor(target, cv.COLOR_BGRA2RGBA)

    source_h, source_w = source.shape[:2]
    target_h, target_w = target.shape[:2]

    ax.imshow(
        source_rgb,
        extent=(0, source_w, source_h, 0),
    )

    ax.imshow(
        target_rgb,
        extent=(source_w, source_w + target_w, target_h, 0),
    )

    details = result.details

    for match, is_inlier in zip(details.matches, details.inlier_mask):
        if inliers_only and not is_inlier:
            continue

        source_pt = details.source_keypoints[match.queryIdx].pt
        target_pt = details.target_keypoints[match.trainIdx].pt

        x1, y1 = source_pt
        x2, y2 = target_pt

        x2 += source_w

        color = "lime" if is_inlier else "red"

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )

        ax.scatter(
            [x1, x2],
            [y1, y2],
            color=color,
            s=8,
        )

    ax.set_xlim(0, source_w + target_w)
    ax.set_ylim(max(source_h, target_h), 0)
    ax.set_aspect("equal")
    ax.axis("off")


pairs = read_pairs()

for pair in pairs:
    source_img = pair.source
    target_img = pair.target

    result = match_images(
        source=source_img,
        target=target_img,
    )

    mutual = result.mutual_matches
    inliers = result.inliers
    outliers = mutual - inliers

    inlier_ratio = inliers / mutual if mutual > 0 else 0.0

    print(f"Inlier ratio: {inlier_ratio:.2%}")

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    draw_matches(
        axes[0],
        source_img,
        target_img,
        result,
    )

    axes[0].set_title(f"All Mutual Matches: {mutual} | Inliers: {inliers} | Outliers: {outliers}")

    draw_matches(
        axes[1],
        source_img,
        target_img,
        result,
        inliers_only=True,
    )

    axes[1].set_title(f"RANSAC Inliers: {inliers} | Inlier Ratio: {inlier_ratio:.2%}")

    plt.tight_layout()


plt.show()
