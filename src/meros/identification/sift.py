from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2 as cv
from kornia.feature.adalam.adalam import AdalamFilter, get_adalam_default_config
from kornia.feature.adalam.core import extract_neighborhood_sets, select_seeds
from kornia.feature.adalam.utils import dist_matrix
import numpy as np
import torch

from meros.external import build_sam2_predictor
from meros.project import Project
from meros.utils import write_image

OUTPUT = Path("data/media/sandbox/sift")

SUBJECTS = {"rogerio": [0, 9], "rafael": [1, 4], "athila": [0]}

MAX_SIZE = 1600
CLAHE_CLIP_LIMIT = 2.0
LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 4.0


@dataclass
class Features:
    name: str
    gray: np.ndarray
    keypoints: list[cv.KeyPoint]
    descriptors: np.ndarray | None


@dataclass
class ComparisonResult:
    matches: list[cv.DMatch]
    global_inliers: np.ndarray
    local_inliers: np.ndarray


# Crops the image to the bounding box of the mask, resizes it if necessary, and extracts SIFT features.
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


def global_consistency(first: Features, second: Features, matches: list[cv.DMatch]) -> np.ndarray:
    retained = np.zeros(len(matches), dtype=bool)

    if len(matches) < 4:
        return retained

    source = np.asarray([first.keypoints[m.queryIdx].pt for m in matches], dtype=np.float32)
    target = np.asarray([second.keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)

    cv.setRNGSeed(0)
    homography, status = cv.findHomography(
        source,
        target,
        cv.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=10000,
        confidence=0.999,
    )

    if homography is not None and status is not None:
        retained = status.ravel().astype(bool)

    return retained


# Applies the AdaLAM local consistency filter to the candidate matches. Returns a boolean array indicating which matches are retained.
def local_consistency(
    first: Features, second: Features, matches: list[cv.DMatch], scores: torch.Tensor
) -> np.ndarray:
    retained = np.zeros(len(matches), dtype=bool)

    config = get_adalam_default_config()
    assert (
        "min_inliers" in config
        and "area_ratio" in config
        and "orientation_difference_threshold" in config
        and "scale_rate_threshold" in config
        and "search_expansion" in config
    )

    config["device"] = torch.device("cpu")

    if len(matches) < config["min_inliers"]:
        return retained

    assert first.descriptors is not None and second.descriptors is not None

    # Remap only the candidate endpoints; AdaLAM builds a quadratic spatial distance matrix.
    source_points = [first.keypoints[m.queryIdx] for m in matches]
    target_points = [second.keypoints[m.trainIdx] for m in matches]
    source = torch.tensor([p.pt for p in source_points], dtype=torch.float32)
    target = torch.tensor([p.pt for p in target_points], dtype=torch.float32)
    angles1 = torch.tensor([p.angle for p in source_points], dtype=torch.float32)
    angles2 = torch.tensor([p.angle for p in target_points], dtype=torch.float32)
    scales1 = torch.tensor([p.size for p in source_points], dtype=torch.float32)
    scales2 = torch.tensor([p.size for p in target_points], dtype=torch.float32)
    indices = torch.arange(len(matches))

    reverse = cv.BFMatcher(cv.NORM_L2).match(
        second.descriptors[[m.trainIdx for m in matches]], first.descriptors
    )
    reverse_indices = {m.queryIdx: m.trainIdx for m in reverse}
    mutual = torch.tensor([reverse_indices[i] == m.queryIdx for i, m in enumerate(matches)])

    # Kornia falls back to ratio-only matches if no neighborhood survives. Do not verify those.
    distances = dist_matrix(source, source)
    radius1 = torch.sqrt(
        torch.tensor(first.gray.size, dtype=torch.float32) / config["area_ratio"] / np.pi
    )
    radius2 = torch.sqrt(
        torch.tensor(second.gray.size, dtype=torch.float32) / config["area_ratio"] / np.pi
    )
    seeds1, seeds2 = select_seeds(distances, radius1, scores, indices, mutual)
    _, sizes, _, _ = extract_neighborhood_sets(
        angles1,
        angles2,
        scales1,
        scales2,
        distances,
        seeds1,
        seeds2,
        source,
        target,
        radius1,
        radius2,
        indices,
        config["orientation_difference_threshold"],
        config["scale_rate_threshold"],
        config["search_expansion"],
        config["min_inliers"],
    )

    if not len(sizes):
        return retained

    torch.manual_seed(0)
    accepted = AdalamFilter(config).filter_matches(
        source,
        target,
        indices,
        scores,
        mnn=mutual,
        im1shape=tuple(first.gray.shape),
        im2shape=tuple(second.gray.shape),
        o1=angles1,
        o2=angles2,
        s1=scales1,
        s2=scales2,
    )
    assert isinstance(accepted, torch.Tensor)
    retained[accepted[:, 0].numpy()] = True

    return retained


# Compares two photos (taken as Features)
def compare(first: Features, second: Features) -> ComparisonResult:
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
    ratios = {
        pair[0].queryIdx: (pair[0].distance / pair[1].distance) ** 2
        for pair in neighbors
        if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance
    }
    scores = torch.tensor([ratios[m.queryIdx] for m in matches], dtype=torch.float32)

    return ComparisonResult(
        matches=matches,
        global_inliers=global_consistency(first, second, matches),
        local_inliers=local_consistency(first, second, matches, scores),
    )


def save_pair(
    first: Features,
    second: Features,
    result: ComparisonResult,
) -> None:
    thumbnails = []

    for photo in (first, second):
        scale = min(1.0, 640 / max(photo.gray.shape))
        thumbnails.append(
            cv.resize(photo.gray, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        )

    left, right = thumbnails

    canvas = np.zeros(
        (max(left.shape[0], right.shape[0]) + 90, left.shape[1] + right.shape[1], 3), dtype=np.uint8
    )

    canvas[90 : 90 + left.shape[0], : left.shape[1]] = cv.cvtColor(left, cv.COLOR_GRAY2BGR)
    canvas[90 : 90 + right.shape[0], left.shape[1] :] = cv.cvtColor(right, cv.COLOR_GRAY2BGR)

    # Highlight local-only correspondences last so the newly retained matches remain visible.
    categories = [
        (~result.global_inliers & ~result.local_inliers, (65, 65, 150), "Neither"),
        (result.global_inliers & ~result.local_inliers, (0, 165, 255), "Global only"),
        (result.global_inliers & result.local_inliers, (50, 220, 50), "Both"),
        (~result.global_inliers & result.local_inliers, (255, 230, 0), "Local only"),
    ]

    for selection, color, _ in categories:
        for match, selected in zip(result.matches, selection, strict=True):
            if not selected:
                continue

            x, y = first.keypoints[match.queryIdx].pt
            u, v = second.keypoints[match.trainIdx].pt
            source = (
                round(x * left.shape[1] / first.gray.shape[1]),
                round(y * left.shape[0] / first.gray.shape[0]) + 90,
            )
            target = (
                round(u * right.shape[1] / second.gray.shape[1]) + left.shape[1],
                round(v * right.shape[0] / second.gray.shape[0]) + 90,
            )
            cv.line(canvas, source, target, color, 1, cv.LINE_AA)
            cv.circle(canvas, source, 2, color, -1, cv.LINE_AA)
            cv.circle(canvas, target, 2, color, -1, cv.LINE_AA)

    count = int(result.global_inliers.sum())
    title = (
        f"{first.name} ({len(first.keypoints)} kp) vs {second.name} ({len(second.keypoints)} kp)"
    )
    detail = (
        f"{len(result.matches)} candidates | Global homography: {count} | "
        f"AdaLAM local affine: {int(result.local_inliers.sum())}"
    )
    cv.putText(
        canvas, title, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv.LINE_AA
    )
    cv.putText(
        canvas, detail, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA
    )

    x = 10

    for selection, color, label in categories:
        text = f"{label}: {int(selection.sum())}"
        cv.putText(canvas, text, (x, 76), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
        x += cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 25
    filename = f"{first.name.replace('/', '_')}_vs_{second.name.replace('/', '_')}.png"
    write_image(OUTPUT / filename, canvas)


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(2)
    cv.setNumThreads(2)

    project = Project.open()
    subject_images = [
        image
        for subject_id, image_ids in SUBJECTS.items()
        for image in project.media.list_subject_images(subject_id)
        if image.image_id in image_ids
    ]

    photos = []

    predictor = build_sam2_predictor(device="cpu")

    # First, extract the SAM2 masks for each photo. This is done in a separate loop to avoid memory fragmentation.
    with torch.inference_mode():
        for photo in subject_images:
            prompt = photo.segmentation

            if prompt is None:
                continue

            name = f"{photo.subject_id}/{photo.image_id}"
            image = project.media.read_subject_image(photo.subject_id, photo.image_id)

            # Each unrelated photo is a separate one-frame sequence.
            with TemporaryDirectory(prefix="meros-sift-") as directory:
                frame = Path(directory) / "00000000.jpg"
                frame.symlink_to(Path(photo.path).resolve())

                state = predictor.init_state(directory)

                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=np.asarray(prompt.points, dtype=np.float32)
                    * np.asarray(image.shape[1::-1], dtype=np.float32),
                    labels=np.asarray(prompt.labels, dtype=np.int32),
                )

                prediction = next(predictor.propagate_in_video(state))

                mask = prediction.masks["1"].astype(np.uint8) * 255

                del state

            photos.append(prepare_photo(name, image, mask))

    del predictor

    # Then, compare each pair of photos and save the results.
    for a, b in combinations(range(len(photos)), 2):
        first, second = photos[a], photos[b]

        result = compare(first, second)

        save_pair(first, second, result)

        print(
            f"{first.name} vs {second.name}: global {int(result.global_inliers.sum())}, "
            f"local {int(result.local_inliers.sum())} / {len(result.matches)} candidates",
            flush=True,
        )


if __name__ == "__main__":
    run()
