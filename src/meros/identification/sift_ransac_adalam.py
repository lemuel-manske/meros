from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
from kornia.feature.adalam.adalam import AdalamFilter, get_adalam_default_config
from kornia.feature.adalam.core import extract_neighborhood_sets, select_seeds
from kornia.feature.adalam.utils import dist_matrix
import numpy as np
import torch

from meros.identification.common import pairs, save_matches
from meros.project import Project
from meros.utils import configure_torch

OUTPUT = Path("data/media/sandbox/sift_ransac_adalam/results.csv")

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
    homography: np.ndarray | None


def prepare_photo(name: str, crop: np.ndarray) -> Features:
    assert crop.ndim == 3 and crop.shape[2] == 4, f"{name}: expected a BGRA crop"

    crop_mask = crop[:, :, 3]
    assert crop_mask.any(), f"{name}: empty crop mask"

    crop = crop[:, :, :3]

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


def global_consistency(
    first: Features, second: Features, matches: list[cv.DMatch]
) -> tuple[np.ndarray | None, np.ndarray]:
    retained = np.zeros(len(matches), dtype=bool)

    if len(matches) < 4:
        return None, retained

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

    return homography, retained


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

    homography, global_inliers = global_consistency(first, second, matches)

    return ComparisonResult(
        matches=matches,
        global_inliers=global_inliers,
        local_inliers=local_consistency(first, second, matches, scores),
        homography=homography,
    )


def save_pair(first: Features, second: Features, result: ComparisonResult) -> None:
    subject_a, image_a = first.name.split("/")
    subject_b, image_b = second.name.split("/")

    global_inliers = int(result.global_inliers.sum())
    local_inliers = int(result.local_inliers.sum())
    both_inliers = int((result.global_inliers & result.local_inliers).sum())

    save_matches(
        OUTPUT,
        method="sift",
        subject_a=subject_a,
        image_a=image_a,
        subject_b=subject_b,
        image_b=image_b,
        status="ok",
        matches=len(result.matches),
        global_inliers=global_inliers,
        local_inliers=local_inliers,
        both_inliers=both_inliers,
    )


def run(project: Project) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    configure_torch()

    for subject_a, subject_b, a, b in pairs(
        project, a_filter="rogerio", b_filter="rogerio", bin_filter=False
    ):
        first = prepare_photo(f"{subject_a._id}/{a._id}", a.crop_im)
        second = prepare_photo(f"{subject_b._id}/{b._id}", b.crop_im)

        result = compare(first, second)

        save_pair(first, second, result)

        print(f"\nAligning {subject_a._id}/{a._id} with {subject_b._id}/{b._id}")


if __name__ == "__main__":
    run(Project.open())
