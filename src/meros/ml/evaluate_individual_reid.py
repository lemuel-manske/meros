"""
Visual audit of proposed Goliath grouper spot correspondences.

The original matcher is retained as a baseline. Local spot constellations
now generate candidates beyond its narrow global alignment gate. Independent
neighborhood votes select one-to-one links; ambiguous and unsupported links
remain visible for human review.

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
NEIGHBOR_TOLERANCE = 0.08
NEIGHBOR_LIMIT = 12
MIN_NEIGHBORS = 2
MIN_SEED_AGREEMENT = 0.40
SEARCH_RADIUS = 0.65
SHAPE_NEIGHBORS = 5
SHAPE_CANDIDATES = 10
VISUAL_CANDIDATES = 5
MIN_PAIR_VOTES = 0.80
MIN_ALTERNATIVE_MARGIN = 0.15
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
    links: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ConstellationResult:
    accepted: tuple[tuple[int, int], ...]
    rejected: tuple[tuple[int, int], ...]
    ambiguous: tuple[tuple[int, int], ...]
    insufficient: tuple[tuple[int, int], ...]
    corrected: int
    new_sources: int
    seeds: int


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
    Match the neighboring constellations after independently centering them.

    A source/target center is only a hypothesis. One-to-one assignments of
    nearby spots supply new pair proposals rather than just validating the
    original global associations.
    """
    radius = NEIGHBOR_RADIUS * target.scale
    tolerance = NEIGHBOR_TOLERANCE * target.scale
    first_distances = np.linalg.norm(projected - projected[i], axis=1)
    second_distances = np.linalg.norm(target.points - target.points[j], axis=1)
    first = [
        k for k in np.argsort(first_distances)
        if k != i and first_distances[k] <= radius
    ][:NEIGHBOR_LIMIT]
    second = [
        k for k in np.argsort(second_distances)
        if k != j and second_distances[k] <= radius
    ][:NEIGHBOR_LIMIT]

    if min(len(first), len(second)) < MIN_NEIGHBORS:
        return None

    first_vectors = projected[first] - projected[i]
    second_vectors = target.points[second] - target.points[j]
    distances = cdist(first_vectors, second_vectors)
    n, m = distances.shape
    costs = np.full((n + m, n + m), 3.0)
    costs[:n, :m] = np.where(
        distances <= tolerance,
        np.minimum(distances / tolerance, 0.99),
        3.0,
    )
    costs[np.arange(n), m + np.arange(n)] = 0.5
    costs[n + np.arange(m), np.arange(m)] = 0.5
    costs[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(costs)
    links = tuple(
        (int(first[a]), int(second[b]))
        for a, b in zip(rows, cols, strict=True)
        if a < n and b < m and distances[a, b] <= tolerance
    )

    return LocalSupport(2 * len(links) / (n + m), len(links), links)


def radial_fingerprints(points: np.ndarray, scale: float) -> np.ndarray:
    """
    Summarize neighboring spot distances without using visual descriptors.
    """
    distances = cdist(points, points) / scale
    np.fill_diagonal(distances, np.inf)
    nearest = np.sort(distances, axis=1)[:, :SHAPE_NEIGHBORS]
    padded = np.pad(
        nearest,
        ((0, 0), (0, SHAPE_NEIGHBORS - nearest.shape[1])),
        constant_values=NEIGHBOR_RADIUS,
    )
    return np.minimum(padded, NEIGHBOR_RADIUS)


def compare_constellations(
    source: Photo,
    target: Photo,
    coarse: SpotMatch,
) -> ConstellationResult:
    """
    Recover local associations outside the original match radius.

    Shortlist anchor hypotheses by local radial shape AND by visual patches.
    Their neighbors vote for one-to-one correspondences. A unique global
    assignment is proposed only when both endpoints prefer it sufficiently.
    No manually verified links participate in the search.
    """
    projected = project(source.points, initial_transform(source, target))
    appearance = cdist(source.features, target.features, metric="cosine")
    geometry = cdist(projected, target.points) / target.scale
    first_shapes = radial_fingerprints(projected, target.scale)
    second_shapes = radial_fingerprints(target.points, target.scale)
    shape_distance = cdist(first_shapes, second_shapes)
    shape_distance += 0.10 * geometry
    old_links = set(coarse.pairs)
    votes = np.zeros_like(appearance)
    seed_count = 0

    for i in range(len(source.points)):
        eligible = np.flatnonzero(geometry[i] <= SEARCH_RADIUS)
        candidates: set[int] = {
            j for a, j in old_links if a == i
        }

        if len(eligible):
            candidates.update(
                eligible[np.argsort(shape_distance[i, eligible])[:SHAPE_CANDIDATES]]
            )
            candidates.update(
                eligible[np.argsort(appearance[i, eligible])[:VISUAL_CANDIDATES]]
            )

        for j in candidates:
            support = local_support(source, target, projected, i, int(j))

            if support is None or support.neighbors_matched < MIN_NEIGHBORS:
                continue

            if support.agreement < MIN_SEED_AGREEMENT:
                continue

            seed_count += 1
            weight = support.agreement
            votes[i, j] += weight

            for first, second in support.links:
                votes[first, second] += weight

    # Appearance is a secondary preference, never a hard gate for a
    # correspondence supported by multiple neighboring constellations.
    votes *= np.clip(1.0 - 0.3 * appearance, 0.50, 1.0)
    eligible = votes >= MIN_PAIR_VOTES
    n, m = votes.shape
    costs = np.zeros((n + m, n + m))
    costs[:n, :m] = np.where(eligible, -votes, 1.0)
    rows, cols = linear_sum_assignment(costs)
    proposed = [
        (int(i), int(j))
        for i, j in zip(rows, cols, strict=True)
        if i < n and j < m and eligible[i, j]
    ]
    accepted, ambiguous = [], []

    for i, j in proposed:
        own = votes[i, j]
        source_second = max(
            (votes[i, k] for k in range(m) if k != j), default=0.0
        )
        target_second = max(
            (votes[k, j] for k in range(n) if k != i), default=0.0
        )

        if own - max(source_second, target_second) >= MIN_ALTERNATIVE_MARGIN * own:
            accepted.append((i, j))
        else:
            ambiguous.append((i, j))

    accepted.sort()
    chosen = {i: j for i, j in accepted}
    proposed_sources = {i for i, _ in proposed}
    rejected, insufficient = [], []

    for i, j in coarse.pairs:
        if i in chosen or i in proposed_sources:
            continue

        if np.max(votes[i]) < MIN_PAIR_VOTES:
            insufficient.append((i, j))
        else:
            rejected.append((i, j))

    corrected = sum(
        i in chosen and chosen[i] != j for i, j in coarse.pairs
    )
    old_sources = {i for i, _ in coarse.pairs}
    new_sources = sum(i not in old_sources for i in chosen)

    return ConstellationResult(
        tuple(accepted), tuple(rejected), tuple(ambiguous),
        tuple(insufficient), corrected, new_sources, seed_count,
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

    for axis, label, photo in zip(axes, (first_label, second_label), (source, target), strict=True):
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
                points[:, 0],
                points[:, 1],
                s=size,
                marker=marker,
                facecolors="none" if marker == "o" else color,
                edgecolors=color if marker == "o" else "none",
                linewidths=1.5,
                zorder=5,
            )

    for i, j in result.accepted[:MAX_LINES]:
        a, b = source.points[i], target.points[j]
        figure.add_artist(
            ConnectionPatch(
                xyA=(float(a[0]), float(a[1])),
                xyB=(float(b[0]), float(b[1])),
                coordsA="data",
                coordsB="data",
                axesA=axes[0],
                axesB=axes[1],
                color="#39f5e8",
                alpha=0.6,
                linewidth=1,
            )
        )

    pair_name = f"{first_label[0]}/{first_label[1]} ↔ {second_label[0]}/{second_label[1]}"
    print(
        f"{pair_name}: original={len(coarse.pairs)}, "
        f"accepted={len(result.accepted)}, corrected={result.corrected}, "
        f"new-sources={result.new_sources}, seeds={result.seeds}, "
        f"rejected={len(result.rejected)}, ambiguous={len(result.ambiguous)}, "
        f"insufficient={len(result.insufficient)}"
    )
    truth = verified_pairs(project, first_label, second_label, source, target)

    if truth:
        proposals = tuple(sorted((*result.accepted, *result.ambiguous)))
        report_verified("original", coarse.pairs, truth)
        report_verified("constellation candidates", proposals, truth)
        report_verified("constellation accepted", result.accepted, truth)

    figure.suptitle(f"{pair_name} | {len(result.accepted)} distinctive constellations")
    figure.text(
        0.5,
        0.035,
        "Cyan: distinctive links | Pink X: unsupported | "
        "Orange: ambiguous | Gray: insufficient | Yellow: annotated spots",
        ha="center",
        fontsize=9,
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
