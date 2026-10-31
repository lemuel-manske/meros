"""
Compare proximity, frozen ResNet18 patch features, and visual+geometric spot matching.

No verified spot correspondences or subject identities are provided to matchers.
Verified links are read only after matching to evaluate proposed pairs.
Torch and torchvision are optional dependencies for this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist

from meros.ml.match_spot_patterns import (
    SpotMatch,
    initial_similarity,
    match_spots,
    project_points,
    verified_pairs,
)
from meros.project import Project


PATCH_SCALE = 0.22
PATCH_SIZE = 128
MAX_VISUAL_DISTANCE = 0.35
GEOMETRIC_RADIUS_RATIO = 0.12
GEOMETRIC_WEIGHT = 0.35
RANSAC_RADIUS_RATIO = 0.06

POSITIVE_PAIRS = (
    (("luana", 0), ("luana", 1)),
    (("rafael", 0), ("rafael", 1)),
    (("rogerio", 0), ("rogerio", 1)),
    (("rogerio", 0), ("rogerio", 9)),
    (("yam", 0), ("yam", 1)),
)
NEGATIVE_PAIRS = (
    (("luana", 0), ("rafael", 1)),
    (("luana", 0), ("rogerio", 1)),
    (("luana", 0), ("yam", 1)),
    (("rafael", 0), ("rogerio", 1)),
    (("rafael", 0), ("yam", 1)),
    (("rogerio", 0), ("yam", 1)),
)


@dataclass(frozen=True)
class SpotPhoto:
    points: np.ndarray
    descriptors: np.ndarray
    eye: tuple[float, float]
    mouth: tuple[float, float]

    @property
    def scale(self) -> float:
        return float(np.linalg.norm(np.asarray(self.mouth) - self.eye))


def create_encoder():
    """
    Load a frozen ImageNet ResNet18. The weights download on the first run.
    """
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder = torch.nn.Sequential(*list(backbone.children())[:-2])
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    return encoder


def extract_descriptors(
    image: np.ndarray,
    points: np.ndarray,
    scale: float,
    encoder,
) -> np.ndarray:
    """
    Embed grayscale contrast-normalized patches centered on manual spots.
    """
    import torch
    import torch.nn.functional as functional

    patch_width = max(16, round(PATCH_SCALE * scale))
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equalizer = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    patches = []

    for x, y in points:
        patch = cv.getRectSubPix(grayscale, (patch_width, patch_width), (float(x), float(y)))
        patch = cv.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv.INTER_AREA)
        patch = equalizer.apply(patch).astype(np.float32) / 255.0
        patches.append(patch)

    if not patches:
        return np.empty((0, 512), dtype=np.float32)

    tensor = torch.from_numpy(np.stack(patches))
    tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    chunks = []

    with torch.inference_mode():
        for batch in tensor.split(32):
            features = encoder(batch)
            features = functional.adaptive_avg_pool2d(features, 1).flatten(1)
            features = functional.normalize(features, dim=1)
            chunks.append(features.numpy())

    return np.concatenate(chunks, axis=0)


def load_photo(
    project: Project,
    label: tuple[str, int],
    encoder,
) -> SpotPhoto | None:
    """
    Read a photo and its manual spot/eye/mouth annotations through Project.
    """
    eye = project.metadata.read_subject_eye(*label)
    mouth = project.metadata.read_subject_mouth(*label)

    if eye is None or mouth is None:
        return None

    points = np.asarray(project.metadata.read_subject_spots(*label), dtype=np.float64)

    if len(points) < 3:
        return None

    scale = float(np.linalg.norm(np.asarray(mouth) - eye.center))

    if scale <= 0:
        return None

    image = project.media.read_subject_image(*label)
    descriptors = extract_descriptors(image, points, scale, encoder)

    return SpotPhoto(points, descriptors, eye.center, mouth)


def assign_candidates(costs: np.ndarray, eligible: np.ndarray) -> tuple[tuple[int, int], ...]:
    """
    Choose one-to-one candidate links ordered by their combined cost.
    """
    first_indices, second_indices = np.where(eligible)
    order = np.argsort(costs[first_indices, second_indices], kind="stable")
    used_first: set[int] = set()
    used_second: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for position in order:
        first = int(first_indices[position])
        second = int(second_indices[position])

        if first in used_first or second in used_second:
            continue

        pairs.append((first, second))
        used_first.add(first)
        used_second.add(second)

    return tuple(pairs)


def visual_pairs(source: SpotPhoto, target: SpotPhoto) -> tuple[tuple[int, int], ...]:
    """
    Match by appearance only, using mutual nearest neighbors.
    """
    distances = cdist(source.descriptors, target.descriptors, metric="cosine")
    nearest_target = np.argmin(distances, axis=1)
    nearest_source = np.argmin(distances, axis=0)

    return tuple(
        (first, int(second))
        for first, second in enumerate(nearest_target)
        if nearest_source[second] == first and distances[first, second] <= MAX_VISUAL_DISTANCE
    )


def geometric_visual_match(source: SpotPhoto, target: SpotPhoto) -> SpotMatch:
    """
    Combine appearance with anatomical projection and refine by affine RANSAC.
    """
    visual_distances = cdist(source.descriptors, target.descriptors, metric="cosine")
    matrix = initial_similarity(source.eye, source.mouth, target.eye, target.mouth)
    radius = GEOMETRIC_RADIUS_RATIO * target.scale

    def propose(candidate: np.ndarray) -> tuple[tuple[int, int], ...]:
        projection_distances = cdist(project_points(source.points, candidate), target.points)
        eligible = (
            (visual_distances <= MAX_VISUAL_DISTANCE)
            & (projection_distances <= radius)
        )
        costs = (
            visual_distances / MAX_VISUAL_DISTANCE
            + GEOMETRIC_WEIGHT * projection_distances / radius
        )

        return assign_candidates(costs, eligible)

    pairs = propose(matrix)

    if len(pairs) >= 3:
        first, second = zip(*pairs, strict=True)

        try:
            affine, inliers = cv.estimateAffine2D(
                source.points[list(first)],
                target.points[list(second)],
                method=cv.RANSAC,
                ransacReprojThreshold=RANSAC_RADIUS_RATIO * target.scale,
            )
        except cv.error:
            affine, inliers = None, None

        if affine is not None and inliers is not None and int(inliers.sum()) >= 3:
            candidate_matrix = np.vstack((affine, [0.0, 0.0, 1.0]))
            candidate_pairs = propose(candidate_matrix)

            if len(candidate_pairs) >= len(pairs):
                matrix = candidate_matrix
                pairs = candidate_pairs

    projected = project_points(source.points, matrix)
    errors = tuple(float(np.linalg.norm(projected[i] - target.points[j])) for i, j in pairs)

    return SpotMatch(matrix=matrix, pairs=pairs, errors=errors)


def report(
    name: str,
    pairs: tuple[tuple[int, int], ...],
    source: SpotPhoto,
    target: SpotPhoto,
    verified: set[tuple[int, int]],
) -> None:
    """
    Summarize proposals and verified recovery, without assuming unlabeled links are false.
    """
    proposals = set(pairs)
    recovered = proposals & verified
    known_first = {i for i, _ in verified}
    known_second = {j for _, j in verified}
    conflicts = {
        (i, j)
        for i, j in proposals - verified
        if i in known_first or j in known_second
    }

    coverage = len(pairs) / min(len(source.points), len(target.points))
    visual_distances = cdist(source.descriptors, target.descriptors, metric="cosine")
    mean_similarity = (
        float(np.mean([1.0 - visual_distances[i, j] for i, j in pairs]))
        if pairs
        else float("nan")
    )
    verified_text = f"{len(recovered)}/{len(verified)}" if verified else "n/a"
    conflict_text = str(len(conflicts)) if verified else "n/a"

    print(
        f"  {name:18} | {len(pairs):3} | {coverage:6.2f} | "
        f"{mean_similarity:7.3f} | {verified_text:7} | {conflict_text:>3}"
    )


def run(project: Project) -> None:
    """
    Evaluate the same three methods on positive and impostor pairs.
    """
    try:
        encoder = create_encoder()
    except ImportError as error:
        raise RuntimeError(
            "This experiment requires torch and torchvision with compatible versions."
        ) from error

    photo_cache: dict[tuple[str, int], SpotPhoto | None] = {}

    for first_label, second_label in (*POSITIVE_PAIRS, *NEGATIVE_PAIRS):
        for label in (first_label, second_label):
            if label not in photo_cache:
                photo_cache[label] = load_photo(project, label, encoder)

        source = photo_cache[first_label]
        target = photo_cache[second_label]

        if source is None or target is None:
            print(f"SKIP {first_label} vs {second_label}: missing image annotations")
            continue

        verified = (
            verified_pairs(project, first_label, second_label, source.points, target.points)
            if first_label[0] == second_label[0]
            else set()
        )
        proximity = match_spots(
            source.points, target.points, source.eye, source.mouth,
            target.eye, target.mouth, radius_ratio=GEOMETRIC_RADIUS_RATIO,
        )
        visual = visual_pairs(source, target)
        combined = geometric_visual_match(source, target)

        print(
            f"\\n{first_label} vs {second_label} "
            f"({'same' if first_label[0] == second_label[0] else 'different'})"
        )
        print("  method             | n   | cover  | mean_sim | verified | conflicts")
        report("proximity", proximity.pairs, source, target, verified)
        report("visual", visual, source, target, verified)
        report("visual + geometry", combined.pairs, source, target, verified)


if __name__ == "__main__":
    run(Project.open())
