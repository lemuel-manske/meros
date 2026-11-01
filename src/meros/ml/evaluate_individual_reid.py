"""
Visual audit of proposed Goliath grouper spot correspondences.

The original matcher proposes a coarse anatomical transform. A local
constellation independently compares ALL annotated neighboring spots and
tests alternative center assignments. Only distinctive, one-to-one matches
are connected; ambiguous and unsupported proposals remain visibly separate.

Local support is diagnostic, not proof of identity. Sparse annotations,
viewpoint changes and longitudinal spot variation can cause false rejections.
Edit PAIRS to inspect other photos; close each figure to see the next pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from meros.project import Project


PAIRS = (
    (("athila", 0), ("rogerio", 1)),
    (("rafael", 3), ("rogerio", 1)),
    (("rafael", 3), ("rafael", 2)),
    (("rogerio", 1), ("rogerio", 4)),
    (("rogerio", 0), ("rogerio", 1)),
    (("rogerio", 11), ("rogerio", 12)),
)

PATCH_SCALE = 0.22
PATCH_SIZE = 128
MAX_VISUAL_DISTANCE = 0.35
GEOMETRIC_RADIUS = 0.12
GEOMETRIC_WEIGHT = 0.35
RANSAC_RADIUS = 0.06
MIN_SPOTS = 4

NEIGHBOR_RADIUS = 0.40
NEIGHBOR_TOLERANCE = 0.045
MIN_NEIGHBORS = 2
MIN_LOCAL_MATCHES = 2
MIN_LOCAL_AGREEMENT = 0.60
MIN_ALTERNATIVE_MARGIN = 0.10
TRUTH_POINT_TOLERANCE = 0.06
MAX_LINES = 20

Label = tuple[str, int]


@dataclass(frozen=True)
class Photo:
    points: np.ndarray
    features: np.ndarray
    eye: tuple[float, float]
    mouth: tuple[float, float]
    image: np.ndarray

    @property
    def scale(self) -> float:
        return float(np.linalg.norm(np.asarray(self.mouth) - self.eye))


@dataclass(frozen=True)
class SpotMatch:
    matrix: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    errors: tuple[float, ...]


@dataclass(frozen=True)
class LocalSupport:
    agreement: float
    neighbors_matched: int


@dataclass(frozen=True)
class ConstellationResult:
    accepted: tuple[tuple[int, int], ...]
    rejected: tuple[tuple[int, int], ...]
    ambiguous: tuple[tuple[int, int], ...]
    insufficient: tuple[tuple[int, int], ...]
    corrected: int


def create_encoder():
    """
    Load frozen ImageNet ResNet18 descriptors; weights download on first use.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from torchvision.models import ResNet18_Weights  # pyright: ignore[reportMissingImports]
    from torchvision.models import resnet18  # pyright: ignore[reportMissingImports]

    network = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder = torch.nn.Sequential(*list(network.children())[:-2])
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    return encoder


def load_photo(project: Project, label: Label, encoder) -> Photo:
    """
    Read images and manual spots/landmarks using project adapters.
    """
    eye = project.metadata.read_subject_eye(*label)
    mouth = project.metadata.read_subject_mouth(*label)
    points = np.asarray(project.metadata.read_subject_spots(*label), dtype=np.float64)

    if eye is None or mouth is None or len(points) < MIN_SPOTS:
        raise ValueError(f"Missing annotations for {label}")

    image = project.media.read_subject_image(*label)
    scale = float(np.linalg.norm(np.asarray(mouth) - eye.center))

    if scale <= 0:
        raise ValueError(f"Invalid eye/mouth scale: {label}")

    import torch  # pyright: ignore[reportMissingImports]
    import torch.nn.functional as functional  # pyright: ignore[reportMissingImports]

    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equalizer = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    width = max(16, round(PATCH_SCALE * scale))
    patches = []

    for x, y in points:
        patch = cv.getRectSubPix(grayscale, (width, width), (float(x), float(y)))
        patch = cv.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv.INTER_AREA)
        patches.append(equalizer.apply(patch).astype(np.float32) / 255.0)

    batch = torch.from_numpy(np.stack(patches)).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    batch = (batch - mean) / std
    features = []

    with torch.inference_mode():
        for part in batch.split(32):
            encoded = functional.adaptive_avg_pool2d(encoder(part), 1).flatten(1)
            features.append(functional.normalize(encoded, dim=1).numpy())

    return Photo(points, np.concatenate(features), eye.center, mouth, image)


def initial_transform(source: Photo, target: Photo) -> np.ndarray:
    """
    Align eye-to-mouth axes before proposing spot associations.
    """
    a = np.asarray(source.mouth) - source.eye
    b = np.asarray(target.mouth) - target.eye
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sine = np.linalg.det(np.stack((a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    linear = rotation * (np.linalg.norm(b) / np.linalg.norm(a))
    transform = np.eye(3)
    transform[:2, :2] = linear
    transform[:2, 2] = np.asarray(target.eye) - linear @ np.asarray(source.eye)
    return transform


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv.perspectiveTransform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)


def match_spots(source: Photo, target: Photo) -> SpotMatch:
    """
    Keep the original visual/geometric proposals and affine RANSAC refinement.
    """
    visual = cdist(source.features, target.features, metric="cosine")
    matrix = initial_transform(source, target)
    radius = GEOMETRIC_RADIUS * target.scale

    def propose(transform: np.ndarray) -> tuple[tuple[int, int], ...]:
        distances = cdist(project(source.points, transform), target.points)
        allowed = (visual <= MAX_VISUAL_DISTANCE) & (distances <= radius)
        costs = visual / MAX_VISUAL_DISTANCE + GEOMETRIC_WEIGHT * distances / radius
        positions = sorted(np.argwhere(allowed), key=lambda ij: costs[ij[0], ij[1]])
        taken_source: set[int] = set()
        taken_target: set[int] = set()
        result = []

        for i, j in positions:
            if int(i) not in taken_source and int(j) not in taken_target:
                result.append((int(i), int(j)))
                taken_source.add(int(i))
                taken_target.add(int(j))

        return tuple(result)

    pairs = propose(matrix)

    if len(pairs) >= 3:
        first, second = zip(*pairs, strict=True)
        affine, inliers = cv.estimateAffine2D(
            source.points[list(first)],
            target.points[list(second)],
            method=cv.RANSAC,
            ransacReprojThreshold=RANSAC_RADIUS * target.scale,
        )

        if affine is not None and inliers is not None and int(inliers.sum()) >= 3:
            refined = np.vstack((affine, [0.0, 0.0, 1.0]))
            new_pairs = propose(refined)

            if len(new_pairs) >= len(pairs):
                matrix, pairs = refined, new_pairs

    predicted = project(source.points, matrix)
    errors = tuple(float(np.linalg.norm(predicted[i] - target.points[j])) for i, j in pairs)
    return SpotMatch(matrix, pairs, errors)


def local_support(
    source: Photo,
    target: Photo,
    projected: np.ndarray,
    i: int,
    j: int,
) -> LocalSupport | None:
    """
    Compare two complete neighboring constellations, independently of links.

    The center is fixed, and other annotated spots can be assigned only once.
    Neither the neighborhood membership nor the assignment uses prior matches.
    """
    radius = NEIGHBOR_RADIUS * target.scale
    tolerance = NEIGHBOR_TOLERANCE * target.scale
    first = np.flatnonzero(
        (np.linalg.norm(projected - projected[i], axis=1) <= radius)
        & (np.arange(len(projected)) != i)
    )
    second = np.flatnonzero(
        (np.linalg.norm(target.points - target.points[j], axis=1) <= radius)
        & (np.arange(len(target.points)) != j)
    )

    if min(len(first), len(second)) < MIN_NEIGHBORS:
        return None

    first_vectors = projected[first] - projected[i]
    second_vectors = target.points[second] - target.points[j]
    distances = cdist(first_vectors, second_vectors)
    n, m = distances.shape
    cost = np.full((n + m, n + m), 3.0)
    cost[:n, :m] = np.where(
        distances <= tolerance, np.minimum(distances / tolerance, 0.99), 3.0
    )
    cost[np.arange(n), m + np.arange(n)] = 0.5
    cost[n + np.arange(m), np.arange(m)] = 0.5
    cost[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    hits = sum(
        1
        for a, b in zip(rows, cols, strict=True)
        if a < n and b < m and distances[a, b] <= tolerance
    )
    return LocalSupport(2 * hits / (n + m), hits)


def compare_constellations(
    source: Photo,
    target: Photo,
    coarse: SpotMatch,
) -> ConstellationResult:
    """
    Jointly select center associations, rejecting near-equivalent alternatives.

    Global alignment restricts candidates, but every neighborhood is rebuilt
    from all annotated spots. Mutual best-center preference and the runner-up
    margin prevent an arbitrary center swap within a similar constellation.
    """
    projected = project(source.points, coarse.matrix)
    geometry = cdist(projected, target.points)
    appearance = cdist(source.features, target.features, metric="cosine")
    eligible = (
        (geometry <= GEOMETRIC_RADIUS * target.scale)
        & (appearance <= MAX_VISUAL_DISTANCE)
    )
    evidence: dict[tuple[int, int], LocalSupport] = {}

    for i, j in np.argwhere(eligible):
        score = local_support(source, target, projected, int(i), int(j))

        if score is not None:
            evidence[(int(i), int(j))] = score

    def choices(index: int, *, source_side: bool) -> list[tuple[tuple[int, int], float]]:
        items = [
            (pair, result.agreement)
            for pair, result in evidence.items()
            if pair[0 if source_side else 1] == index
        ]
        return sorted(items, key=lambda item: -item[1])

    by_source = {i: choices(i, source_side=True) for i in range(len(source.points))}
    by_target = {j: choices(j, source_side=False) for j in range(len(target.points))}
    accepted = []

    for pair, result in evidence.items():
        if result.neighbors_matched < MIN_LOCAL_MATCHES:
            continue

        if result.agreement < MIN_LOCAL_AGREEMENT:
            continue

        first_rank = by_source[pair[0]]
        second_rank = by_target[pair[1]]

        if first_rank[0][0] != pair or second_rank[0][0] != pair:
            continue

        first_margin = (
            result.agreement - first_rank[1][1] if len(first_rank) > 1 else 1.0
        )
        second_margin = (
            result.agreement - second_rank[1][1] if len(second_rank) > 1 else 1.0
        )

        if min(first_margin, second_margin) >= MIN_ALTERNATIVE_MARGIN:
            accepted.append(pair)

    accepted.sort()
    accepted_by_source = {i: j for i, j in accepted}
    rejected, ambiguous, insufficient = [], [], []

    for i, j in coarse.pairs:
        if i in accepted_by_source:
            continue

        options = by_source[i]

        if not options:
            insufficient.append((i, j))
        elif options[0][1] >= MIN_LOCAL_AGREEMENT:
            ambiguous.append((i, j))
        else:
            rejected.append((i, j))

    corrected = sum(
        i in accepted_by_source and accepted_by_source[i] != j
        for i, j in coarse.pairs
    )

    return ConstellationResult(
        tuple(accepted), tuple(rejected), tuple(ambiguous),
        tuple(insufficient), corrected
    )


def verified_pairs(
    project: Project,
    first: Label,
    second: Label,
    source: Photo,
    target: Photo,
) -> dict[int, int]:
    """
    Attach existing manually verified coordinate pairs to annotated spot IDs.

    A missing or empty reference is never interpreted as negative evidence.
    """
    if first[0] != second[0]:
        return {}

    reverse = first[1] > second[1]
    a, b = sorted((first[1], second[1]))
    records = project.metadata.read_subject_spot_correspondences(first[0], a, b)
    truth: dict[int, int] = {}
    used_targets = set()

    for record in records:
        left = record["second"] if reverse else record["first"]
        right = record["first"] if reverse else record["second"]
        source_distances = np.linalg.norm(source.points - left, axis=1)
        target_distances = np.linalg.norm(target.points - right, axis=1)
        i, j = int(np.argmin(source_distances)), int(np.argmin(target_distances))

        if source_distances[i] > TRUTH_POINT_TOLERANCE * source.scale:
            continue

        if target_distances[j] > TRUTH_POINT_TOLERANCE * target.scale:
            continue

        if i not in truth and j not in used_targets:
            truth[i] = j
            used_targets.add(j)

    if records:
        print(
            f"  Verified correspondence labels: {len(truth)}/{len(records)} "
            "mapped to spot annotations"
        )

    return truth


def report_verified(
    name: str,
    pairs: tuple[tuple[int, int], ...],
    truth: dict[int, int],
) -> None:
    """
    Count confirmed links, wrong links on reviewed source spots and recall.

    Predictions on unreviewed source spots cannot be classified as false.
    """
    proposed = dict(pairs)
    correct = sum(proposed.get(i) == j for i, j in truth.items())
    wrong = sum(i in truth and proposed[i] != truth[i] for i in proposed)
    print(
        f"  {name}: verified={correct}/{len(truth)}, "
        f"wrong-on-verified-source={wrong}, "
        f"unverified-predictions={len(proposed) - correct - wrong}"
    )


def draw_pair(
    project: Project,
    first_label: Label,
    second_label: Label,
    source: Photo,
    target: Photo,
) -> None:
    """
    Show jointly supported links, rejected links and ambiguous center choices.
    """
    coarse = match_spots(source, target)
    result = compare_constellations(source, target, coarse)
    figure, axes = plt.subplots(1, 2, figsize=(17, 8))

    for axis, label, photo in zip(
        axes, (first_label, second_label), (source, target), strict=True
    ):
        axis.imshow(cv.cvtColor(photo.image, cv.COLOR_BGR2RGB))
        axis.scatter(photo.points[:, 0], photo.points[:, 1], s=12, c="#ffcc59")
        axis.scatter(*photo.eye, s=115, c="white", marker="+", linewidths=2)
        axis.scatter(*photo.mouth, s=55, c="white", marker="s", facecolors="none")
        axis.set_title(f"{label[0]}/{label[1]}")
        axis.set_axis_off()

    for links, color, marker, size in (
        (result.rejected, "#ff4182", "x", 62),
        (result.insufficient, "#b4b4b4", "o", 56),
        (result.ambiguous, "#ffab36", "o", 56),
        (result.accepted, "#39f5e8", "o", 32),
    ):
        if not links:
            continue

        for axis, photo, side in zip(axes, (source, target), (0, 1), strict=True):
            points = photo.points[[link[side] for link in links]]
            axis.scatter(
                points[:, 0], points[:, 1], s=size, marker=marker,
                facecolors="none" if marker == "o" else color,
                edgecolors=color if marker == "o" else "none",
                linewidths=1.5, zorder=5,
            )

    for i, j in result.accepted[:MAX_LINES]:
        a, b = source.points[i], target.points[j]
        figure.add_artist(
            ConnectionPatch(
                xyA=(float(a[0]), float(a[1])),
                xyB=(float(b[0]), float(b[1])),
                coordsA="data", coordsB="data",
                axesA=axes[0], axesB=axes[1],
                color="#39f5e8", alpha=0.6, linewidth=1,
            )
        )

    pair_name = (
        f"{first_label[0]}/{first_label[1]} ↔ "
        f"{second_label[0]}/{second_label[1]}"
    )
    print(
        f"{pair_name}: original={len(coarse.pairs)}, "
        f"accepted={len(result.accepted)}, corrected={result.corrected}, "
        f"rejected={len(result.rejected)}, ambiguous={len(result.ambiguous)}, "
        f"insufficient={len(result.insufficient)}"
    )
    truth = verified_pairs(project, first_label, second_label, source, target)

    if truth:
        report_verified("original", coarse.pairs, truth)
        report_verified("constellation", result.accepted, truth)

    figure.suptitle(
        f"{pair_name} | {len(result.accepted)} distinctive constellations"
    )
    figure.text(
        0.5, 0.035,
        "Cyan: distinctive links | Pink X: unsupported | "
        "Orange: ambiguous | Gray: insufficient | Yellow: annotated spots",
        ha="center", fontsize=9,
    )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10, wspace=0.04)
    plt.show(block=True)
    plt.close(figure)


def run(project: Project) -> None:
    """
    Always visualize each selected pair; no modes, flags or exported artifacts.
    """
    encoder = create_encoder()
    photos: dict[Label, Photo] = {}

    for first, second in PAIRS:
        for label in (first, second):
            if label not in photos:
                photos[label] = load_photo(project, label, encoder)

        draw_pair(project, first, second, photos[first], photos[second])


if __name__ == "__main__":
    run(Project.open())
