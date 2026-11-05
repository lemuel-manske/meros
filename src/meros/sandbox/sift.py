from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2 as cv
import numpy as np
import torch

from meros.external import build_sam2_predictor
from meros.project import Project
from meros.utils import write_image

OUTPUT = Path("data/media/sandbox/sift")

MAX_SIZE = 1600
CLAHE_CLIP_LIMIT = 2.0
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 4.0


# Normalized coordinates: three fish points followed by two background points.
PROMPTS = {
    "rogerio/0": [(0.5, 0.45), (0.35, 0.55), (0.8, 0.6), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/1": [(0.55, 0.45), (0.30, 0.60), (0.85, 0.70), (0.10, 0.10), (0.10, 0.90)],
    "rogerio/2": [(0.55, 0.45), (0.35, 0.55), (0.8, 0.65), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/3": [(0.55, 0.45), (0.3, 0.5), (0.8, 0.7), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/4": [(0.55, 0.45), (0.3, 0.65), (0.75, 0.65), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/5": [(0.55, 0.45), (0.3, 0.6), (0.8, 0.65), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/6": [(0.55, 0.45), (0.45, 0.65), (0.75, 0.75), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/7": [(0.55, 0.45), (0.4, 0.6), (0.8, 0.7), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/8": [(0.55, 0.45), (0.3, 0.7), (0.8, 0.6), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/9": [(0.65, 0.5), (0.4, 0.65), (0.8, 0.7), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/10": [(0.6, 0.5), (0.4, 0.6), (0.8, 0.6), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/11": [(0.6, 0.5), (0.35, 0.5), (0.8, 0.7), (0.05, 0.1), (0.05, 0.9)],
    "rogerio/12": [(0.55, 0.5), (0.3, 0.65), (0.8, 0.65), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/13": [(0.5, 0.5), (0.35, 0.5), (0.6, 0.7), (0.1, 0.1), (0.1, 0.9)],
    "rogerio/14": [(0.55, 0.50), (0.30, 0.65), (0.85, 0.60), (0.10, 0.10), (0.10, 0.90)],
    "rogerio/15": [(0.6, 0.5), (0.35, 0.65), (0.8, 0.6), (0.1, 0.1), (0.1, 0.9)],
    "rafael/0": [(0.55, 0.5), (0.3, 0.6), (0.8, 0.7), (0.05, 0.1), (0.05, 0.9)],
    "rafael/1": [(0.5, 0.5), (0.3, 0.8), (0.65, 0.4), (0.05, 0.1), (0.95, 0.9)],
    "rafael/2": [(0.55, 0.45), (0.3, 0.6), (0.8, 0.65), (0.1, 0.1), (0.1, 0.9)],
    "rafael/3": [(0.55, 0.45), (0.3, 0.55), (0.8, 0.65), (0.05, 0.1), (0.05, 0.9)],
    "rafael/4": [(0.55, 0.45), (0.25, 0.65), (0.85, 0.70), (0.10, 0.10), (0.10, 0.95)],
}


@dataclass
class Features:
    name: str
    gray: np.ndarray
    keypoints: list[cv.KeyPoint]
    descriptors: np.ndarray | None


def prepare_photo(name: str, image: np.ndarray, mask: np.ndarray) -> Features:
    ys, xs = np.where(mask)

    assert len(xs), f"{name}: empty SAM2 mask"

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1

    crop = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    scale = min(1.0, MAX_SIZE / max(crop.shape[:2]))

    if scale < 1:
        size = (round(crop.shape[1] * scale), round(crop.shape[0] * scale))
        crop = cv.resize(crop, size, interpolation=cv.INTER_AREA)
        crop_mask = cv.resize(crop_mask, size, interpolation=cv.INTER_NEAREST)

    gray = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8)).apply(
        cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    )

    keypoints, descriptors = cv.SIFT.create().detectAndCompute(gray, crop_mask)

    # Do not let silhouette edges or background enter descriptor neighborhoods.
    distance = cv.distanceTransform(
        cv.copyMakeBorder(crop_mask, 1, 1, 1, 1, cv.BORDER_CONSTANT),
        cv.DIST_L2,
        cv.DIST_MASK_PRECISE,
    )[1:-1, 1:-1]

    retained = [
        i
        for i, point in enumerate(keypoints)
        if distance[round(point.pt[1]), round(point.pt[0])] > 6 * point.size
    ]

    keypoints = [keypoints[i] for i in retained]
    descriptors = descriptors[retained] if descriptors is not None and retained else None

    gray[crop_mask == 0] = 0

    return Features(name, gray, keypoints, descriptors)


def compare(first: Features, second: Features) -> tuple[list[cv.DMatch], np.ndarray, bool]:
    cv.setRNGSeed(0)
    neighbors = []

    if first.descriptors is not None and second.descriptors is not None:
        neighbors = cv.BFMatcher(cv.NORM_L2).knnMatch(first.descriptors, second.descriptors, k=2)

    matches = [
        pair[0]
        for pair in neighbors
        if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance
    ]
    unique = {}

    for match in sorted(matches, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)

    matches = list(unique.values())
    source = np.asarray([first.keypoints[m.queryIdx].pt for m in matches], dtype=np.float32)
    target = np.asarray([second.keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)
    inliers = np.zeros(len(matches), dtype=bool)
    verified = False

    if len(matches) >= 4:
        homography, status = cv.findHomography(
            source,
            target,
            cv.RANSAC,
            RANSAC_THRESHOLD,
            maxIters=10000,
            confidence=0.999,
        )
        verified = homography is not None and status is not None

        if verified:
            inliers = status.ravel().astype(bool)

    return matches, inliers, verified


def save_pair(
    first: Features,
    second: Features,
    matches: list[cv.DMatch],
    inliers: np.ndarray,
    verified: bool,
) -> None:
    thumbnails = []

    for photo in (first, second):
        scale = min(1.0, 640 / max(photo.gray.shape))
        thumbnails.append(
            cv.resize(photo.gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        )

    left, right = thumbnails

    canvas = np.zeros(
        (max(left.shape[0], right.shape[0]) + 65, left.shape[1] + right.shape[1], 3), dtype=np.uint8
    )

    canvas[65 : 65 + left.shape[0], : left.shape[1]] = cv.cvtColor(left, cv.COLOR_GRAY2BGR)
    canvas[65 : 65 + right.shape[0], left.shape[1] :] = cv.cvtColor(right, cv.COLOR_GRAY2BGR)

    # Draw outliers first so they do not obscure the inliers.
    for keep, color in ((False, (70, 70, 190)), (True, (50, 230, 50))):
        for match, inlier in zip(matches, inliers, strict=True):
            if bool(inlier) != keep:
                continue

            x, y = first.keypoints[match.queryIdx].pt
            u, v = second.keypoints[match.trainIdx].pt
            source = (
                round(x * left.shape[1] / first.gray.shape[1]),
                round(y * left.shape[0] / first.gray.shape[0]) + 65,
            )
            target = (
                round(u * right.shape[1] / second.gray.shape[1]) + left.shape[1],
                round(v * right.shape[0] / second.gray.shape[0]) + 65,
            )
            cv.line(canvas, source, target, color, 1, cv.LINE_AA)
            cv.circle(canvas, source, 2, color, -1, cv.LINE_AA)
            cv.circle(canvas, target, 2, color, -1, cv.LINE_AA)

    count = int(inliers.sum())
    title = (
        f"{first.name} ({len(first.keypoints)} kp) vs {second.name} ({len(second.keypoints)} kp)"
    )
    detail = (
        f"{count} inliers (green) | {len(matches) - count} outliers (red) | "
        f"{count / len(matches):.1%} inliers"
        if verified and matches
        else f"{len(matches)} unverified matches: no homography"
    )
    cv.putText(
        canvas, title, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv.LINE_AA
    )
    cv.putText(
        canvas, detail, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA
    )
    filename = f"{first.name.replace('/', '_')}_vs_{second.name.replace('/', '_')}.png"
    write_image(OUTPUT / filename, canvas)


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(2)
    cv.setNumThreads(2)

    project = Project.open()

    photos = []

    predictor = build_sam2_predictor(device="cpu")

    # First, extract the SAM2 masks for each photo. This is done in a separate loop to avoid memory fragmentation.
    with torch.inference_mode():
        for name, prompts in PROMPTS.items():
            subject, image_id = name.split("/")

            image = project.media.read_subject_image(subject, int(image_id))

            # Each unrelated photo is a separate one-frame sequence.
            with TemporaryDirectory(prefix="meros-sift-") as directory:
                frame = Path(directory) / "00000000.jpg"
                frame.symlink_to(project.media.subject_image_path(subject, int(image_id)).resolve())

                state = predictor.init_state(directory)

                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=np.asarray(prompts, dtype=np.float32)
                    * np.asarray(image.shape[1::-1], dtype=np.float32),
                    labels=np.array([1, 1, 1, 0, 0], dtype=np.int32),
                )

                prediction = next(predictor.propagate_in_video(state))

                mask = prediction.masks["1"].astype(np.uint8) * 255

                del state

            photos.append(prepare_photo(name, image, mask))

    del predictor

    # Then, compare each pair of photos and save the results.
    for a, b in combinations(range(len(photos)), 2):
        first, second = photos[a], photos[b]

        matches, inliers, verified = compare(first, second)

        save_pair(first, second, matches, inliers, verified)

        print(
            f"{first.name} vs {second.name}: {int(inliers.sum())}/{len(matches)} inliers",
            flush=True,
        )


if __name__ == "__main__":
    run()
