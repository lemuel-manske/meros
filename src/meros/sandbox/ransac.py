from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes

from src.meros.project import Project


SIFT_FEATURES = 4000
DEFAULT_RATIO = 0.9
DEFAULT_ERODE = 7

RANSAC_MIN_MATCHES = 8
RANSAC_ERROR = 3.0
RANSAC_MAX_ITERS = 5000
RANSAC_CONFIDENCE = 0.999


@dataclass(frozen=True)
class MatchDetails:
    source_keypoints: list[cv.KeyPoint]
    target_keypoints: list[cv.KeyPoint]

    matches: list[cv.DMatch]
    inlier_mask: np.ndarray


@dataclass(frozen=True)
class MatchStats:
    # how many points of interest?
    source_keypoints: int
    target_keypoints: int

    # A -> B
    forward_good: int
    # B -> A
    backward_good: int

    # forward and backward :: A -> B, B -> A
    mutual_matches: int

    inliers: int
    ransac_attempted: bool

    details: MatchDetails


def match_images(
    source: np.ndarray,
    target: np.ndarray,
    ratio: float = DEFAULT_RATIO,
    erode: int = DEFAULT_ERODE,
) -> MatchStats:
    sift = cv.SIFT.create(nfeatures=SIFT_FEATURES)

    features = []

    for crop in (source, target):
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 4:
            raise ValueError("Expected uint8 BGRA composites.")

        gray = cv.cvtColor(crop[:, :, :3], cv.COLOR_BGR2GRAY)

        mask = (crop[:, :, 3] == 255).astype(np.uint8) * 255

        if erode > 0:
            mask = cv.erode(mask, np.ones((erode, erode), dtype=np.uint8))

        features.append(sift.detectAndCompute(gray, mask))

    (source_keys, source_descriptors), (target_keys, target_descriptors) = features

    matches = []

    forward_good = []
    backward_good = []

    if source_descriptors is not None and target_descriptors is not None:
        matcher = cv.BFMatcher()

        forward = matcher.knnMatch(source_descriptors, target_descriptors, k=2)

        backward = matcher.knnMatch(target_descriptors, source_descriptors, k=2)

        forward_good = [
            pair[0]
            for pair in forward
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        ]

        backward_good = [
            pair[0]
            for pair in backward
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        ]

        reverse = {match.queryIdx: match.trainIdx for match in backward_good}

        matches = [match for match in forward_good if reverse.get(match.trainIdx) == match.queryIdx]

    keep = np.zeros(len(matches), dtype=bool)

    if len(matches) >= RANSAC_MIN_MATCHES:
        source_points = np.asarray(
            [source_keys[match.queryIdx].pt for match in matches], dtype=np.float32
        )

        target_points = np.asarray(
            [target_keys[match.trainIdx].pt for match in matches], dtype=np.float32
        )

        cv.setRNGSeed(0)

        matrix, inliers = cv.estimateAffine2D(
            source_points,
            target_points,
            method=cv.RANSAC,
            ransacReprojThreshold=RANSAC_ERROR,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
        )

        if matrix is not None and inliers is not None and (np.linalg.det(matrix[:, :2]) > 0):
            keep = inliers.ravel().astype(bool)

    details = MatchDetails(
        source_keypoints=source_keys,
        target_keypoints=target_keys,
        matches=matches,
        inlier_mask=keep,
    )

    stats = MatchStats(
        details=details,
        source_keypoints=len(source_keys),
        target_keypoints=len(target_keys),
        forward_good=len(forward_good),
        backward_good=len(backward_good),
        mutual_matches=len(matches),
        inliers=int(keep.sum()),
        ransac_attempted=len(matches) >= RANSAC_MIN_MATCHES,
    )

    return stats


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


if __name__ == "__main__":
    project = Project.open()

    for vid in project.metadata.read_videos():
        individuals = vid.individuals

        annotations = vid.matching_annotations(
            viewpoint="frontal",
            min_quality=3,
        )

        for annotation in annotations:
            source_path = project.media.masked_crop_path(
                vid.video_id,
                annotation.track_id,
                annotation.start_frame,
            )

            target_path = project.media.masked_crop_path(
                vid.video_id,
                annotation.track_id,
                annotation.end_frame,
            )

            source_img = cv.imread(source_path, cv.IMREAD_UNCHANGED)
            target_img = cv.imread(target_path, cv.IMREAD_UNCHANGED)

            assert source_img is not None, f"Failed to read source image: {source_path}"
            assert target_img is not None, f"Failed to read target image: {target_path}"

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

            axes[0].set_title(
                f"All Mutual Matches: {mutual} | Inliers: {inliers} | Outliers: {outliers}"
            )

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
