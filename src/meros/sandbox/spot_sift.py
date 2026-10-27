"""Visual comparison of ordinary SIFT and spot-filtered SIFT."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from meros.ml.spot_sift import extract_spot_sift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path, default=Path("spot-sift-comparison.png"))
    args = parser.parse_args()

    image = cv.imread(str(args.image), cv.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(args.image)

    mask = None

    if args.mask is not None:
        mask = cv.imread(str(args.mask), cv.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(args.mask)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(nfeatures=2500)
    baseline, _ = sift.detectAndCompute(gray, mask)
    result = extract_spot_sift(image, mask)

    before = cv.drawKeypoints(
        image, baseline, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    after = cv.drawKeypoints(
        image, result.keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    for x, y in result.spots:
        cv.circle(after, (round(float(x)), round(float(y))), 3, (0, 255, 255), 1)

    preview = np.hstack((before, after))

    if not cv.imwrite(str(args.output), preview):
        raise OSError(f"Could not write {args.output}")

    print(f"SIFT: {len(baseline)} keypoints")
    print(f"Spot candidates: {len(result.spots)}")
    print(f"Filtered SIFT: {len(result.keypoints)} keypoints")
    print(f"Preview: {args.output}")


if __name__ == "__main__":
    main()
