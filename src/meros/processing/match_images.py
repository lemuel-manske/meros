import cv2 as cv
import numpy as np

from dataclasses import dataclass

SIFT_FEATURES = 4000
DEFAULT_RATIO = 0.9
DEFAULT_ERODE = 7

RANSAC_MIN_MATCHES = 8
RANSAC_ERROR = 3.0
RANSAC_MAX_ITERS = 5000
RANSAC_CONFIDENCE = 0.999


@dataclass(frozen=True)
class MatchStats:
    source_keypoints: int
    target_keypoints: int

    forward_good: int
    backward_good: int
    mutual_matches: int

    inliers: int
    ransac_attempted: bool


def match_images(
    source: np.ndarray,
    target: np.ndarray,
    ratio: float = DEFAULT_RATIO,
    erode: int = DEFAULT_ERODE,
    *,
    draw: bool = False,
) -> tuple[np.ndarray | None, MatchStats]:
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

    canvas = None

    if draw:
        canvas = np.zeros(
            (max(source.shape[0], target.shape[0]), source.shape[1] + target.shape[1], 3),
            dtype=np.uint8,
        )

        canvas = cv.drawMatches(
            source[:, :, :3],
            source_keys,
            target[:, :, :3],
            target_keys,
            matches,
            canvas,
            matchColor=(100, 100, 100),
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        for match, inlier in zip(matches, keep):
            if not inlier:
                continue

            left = tuple((int(round(value)) for value in source_keys[match.queryIdx].pt))

            x, y = target_keys[match.trainIdx].pt

            right = (int(round(x)) + source.shape[1], int(round(y)))

            cv.line(canvas, left, right, (0, 255, 0), 1, cv.LINE_AA)

            cv.circle(canvas, left, 3, (0, 255, 0), 1)

            cv.circle(canvas, right, 3, (0, 255, 0), 1)

    stats = MatchStats(
        source_keypoints=len(source_keys),
        target_keypoints=len(target_keys),
        forward_good=len(forward_good),
        backward_good=len(backward_good),
        mutual_matches=len(matches),
        inliers=int(keep.sum()),
        ransac_attempted=len(matches) >= RANSAC_MIN_MATCHES,
    )

    return (canvas, stats)
