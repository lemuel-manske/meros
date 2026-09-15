import cv2 as cv
import numpy as np

from itertools import combinations
from pathlib import Path

from src.artifacts import (
    path_to_composite,
    path_to_composite_matches,
    path_to_enhanced_composite,
)

from src.meros.dataset import individuals


def match_composites(
    source: np.ndarray,
    target: np.ndarray,
    ratio: float = 0.75,
) -> tuple[np.ndarray, int, int]:
    sift = cv.SIFT.create(nfeatures=4000)
    features = []

    for crop in (source, target):
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 4:
            raise ValueError("Expected uint8 BGRA composites.")

        # keep identical processing for both variants; do not apply clahe again.
        gray = cv.cvtColor(crop[:, :, :3], cv.COLOR_BGR2GRAY)
        mask = (crop[:, :, 3] == 255).astype(np.uint8) * 255
        mask = cv.erode(mask, np.ones((21, 21), dtype=np.uint8))
        features.append(sift.detectAndCompute(gray, mask))

    (source_keys, source_descriptors), (target_keys, target_descriptors) = features
    matches = []

    if source_descriptors is not None and target_descriptors is not None:
        matcher = cv.BFMatcher()
        forward = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
        backward = matcher.knnMatch(target_descriptors, source_descriptors, k=2)
        reverse = {
            pair[0].queryIdx: pair[0].trainIdx
            for pair in backward
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
        }
        matches = [
            pair[0] for pair in forward
            if len(pair) == 2
            and pair[0].distance < ratio * pair[1].distance
            and reverse.get(pair[0].trainIdx) == pair[0].queryIdx
        ]

    keep = np.zeros(len(matches), dtype=bool)

    # enough correspondences to inspect geometry beyond the minimal affine fit.
    if len(matches) >= 8:
        source_points = np.asarray([source_keys[m.queryIdx].pt for m in matches], dtype=np.float32)
        target_points = np.asarray([target_keys[m.trainIdx].pt for m in matches], dtype=np.float32)

        cv.setRNGSeed(0)
        matrix, inliers = cv.estimateAffine2D(
            source_points, target_points, method=cv.RANSAC,
            ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999,
        )

        if matrix is not None and inliers is not None and np.linalg.det(matrix[:, :2]) > 0:
            keep = inliers.ravel().astype(bool)

    # gray lines are tentative; green lines agree with the fitted geometry.
    canvas = np.zeros((max(source.shape[0], target.shape[0]), source.shape[1] + target.shape[1], 3), dtype=np.uint8)
    canvas = cv.drawMatches(
        source[:, :, :3], source_keys, target[:, :, :3], target_keys, matches, canvas,
        matchColor=(100, 100, 100), flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    for match, inlier in zip(matches, keep):
        if not inlier:
            continue

        left = tuple(int(round(v)) for v in source_keys[match.queryIdx].pt)
        x, y = target_keys[match.trainIdx].pt
        right = (int(round(x)) + source.shape[1], int(round(y)))
        cv.line(canvas, left, right, (0, 255, 0), 1, cv.LINE_AA)
        cv.circle(canvas, left, 3, (0, 255, 0), 1)
        cv.circle(canvas, right, 3, (0, 255, 0), 1)

    return canvas, len(matches), int(keep.sum())


if __name__ == "__main__":
    for individual in individuals:
        for source, target in combinations(individual.tracks, 2):
            if source.video_id == target.video_id:
                continue

            for enhanced in (False, True):
                path_to_input = path_to_enhanced_composite if enhanced else path_to_composite
                crops = []

                for track in (source, target):
                    path = path_to_input(track.video_id, track.track_id)
                    crop = cv.imread(path, cv.IMREAD_UNCHANGED)

                    if crop is None:
                        raise ValueError(f"Could not read composite: {path}.")

                    crops.append(crop)

                canvas, matches, inliers = match_composites(*crops)
                output = Path(path_to_composite_matches(
                    source.video_id, source.track_id,
                    target.video_id, target.track_id, enhanced,
                ))
                output.parent.mkdir(parents=True, exist_ok=True)

                if not cv.imwrite(str(output), canvas):
                    raise RuntimeError(f"Could not save {output}.")

                print(f"{output}: {matches} tentative matches, {inliers} affine inliers.")
