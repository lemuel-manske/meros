"""
Visual audit of proposed Goliath grouper spot correspondences.

A match is only a geometric/visual proposal. Draw a cyan connection only when
nearby annotated spots also preserve their relative geometry. Show inconsistent
proposals as magenta crosses and unassessable proposals as orange circles.

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
from scipy.spatial.distance import cdist

from meros.project import Project


PAIRS = (
    (("athila", 0), ("rogerio", 1)),
    (("rafael", 3), ("rogerio", 1)),
    (("rafael", 3), ("rafael", 2)),
    (("rogerio", 1), ("rogerio", 4)),
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
NEAREST_NEIGHBORS = 6
MIN_NEIGHBORS = 2
MIN_LOCAL_MATCHES = 2
MIN_LOCAL_AGREEMENT = 0.70
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


def neighborhood_support(
    source: Photo,
    target: Photo,
    match: SpotMatch,
) -> list[LocalSupport | None]:
    """
    Check whether each proposed spot link agrees with its linked neighbors.

    Unlike free matching between local spot clouds, this preserves neighbor
    identities: an edge (i -> j) can support another edge (k -> l) only if
    both relative displacement vectors agree. A displaced link in a repeated
    constellation therefore cannot borrow support by relabeling the neighbors.

    Compare the union of the nearest matched neighbors on both photos so
    that one-sided changes in local topology count against agreement.
    """
    if not match.pairs:
        return []

    first, second = zip(*match.pairs, strict=True)
    source_points = project(source.points, match.matrix)[list(first)]
    target_points = target.points[list(second)]
    radius = NEIGHBOR_RADIUS * target.scale
    tolerance = NEIGHBOR_TOLERANCE * target.scale
    output: list[LocalSupport | None] = []

    for index, (anchor_source, anchor_target) in enumerate(
        zip(source_points, target_points, strict=True)
    ):
        source_vectors = source_points - anchor_source
        target_vectors = target_points - anchor_target
        source_distances = np.linalg.norm(source_vectors, axis=1)
        target_distances = np.linalg.norm(target_vectors, axis=1)

        neighbors = np.zeros(len(match.pairs), dtype=bool)
        neighbors[np.argsort(source_distances)[1 : NEAREST_NEIGHBORS + 1]] = True
        neighbors[np.argsort(target_distances)[1 : NEAREST_NEIGHBORS + 1]] = True
        neighbors[index] = False
        neighbors &= (source_distances <= radius) | (target_distances <= radius)

        count = int(neighbors.sum())

        if count < MIN_NEIGHBORS:
            output.append(None)
            continue

        displacement_error = np.linalg.norm(source_vectors - target_vectors, axis=1)
        consistent = (
            neighbors
            & (source_distances <= radius)
            & (target_distances <= radius)
            & (displacement_error <= tolerance)
        )
        hits = int(consistent.sum())
        output.append(LocalSupport(hits / count, hits))

    return output


def draw_pair(
    first_label: Label,
    second_label: Label,
    source: Photo,
    target: Photo,
) -> None:
    """
    Connect ONLY locally supported pairs. Mark the rest without linking.
    """
    match = match_spots(source, target)
    support = neighborhood_support(source, target, match)
    accepted = [
        k
        for k, info in enumerate(support)
        if info is not None
        and info.neighbors_matched >= MIN_LOCAL_MATCHES
        and info.agreement >= MIN_LOCAL_AGREEMENT
    ]
    rejected = [k for k, info in enumerate(support) if info is not None and k not in accepted]
    unknown = [k for k, info in enumerate(support) if info is None]
    figure, axes = plt.subplots(1, 2, figsize=(17, 8))

    for axis, label, photo in zip(axes, (first_label, second_label), (source, target), strict=True):
        axis.imshow(cv.cvtColor(photo.image, cv.COLOR_BGR2RGB))
        axis.scatter(photo.points[:, 0], photo.points[:, 1], s=13, c="#ffcc59")
        axis.scatter(*photo.eye, s=115, c="white", marker="+", linewidths=2)
        axis.scatter(*photo.mouth, s=55, c="white", marker="s", facecolors="none")
        axis.set_title(f"{label[0]}/{label[1]}")
        axis.set_axis_off()

    for positions, color, marker, size in (
        (rejected, "#ff4182", "x", 65),
        (unknown, "#ffab36", "o", 65),
        (accepted, "#39f5e8", "o", 34),
    ):
        if not positions:
            continue

        for axis, photo, side in zip(axes, (source, target), (0, 1), strict=True):
            indices = [match.pairs[k][side] for k in positions]
            points = photo.points[indices]
            axis.scatter(
                points[:, 0],
                points[:, 1],
                s=size,
                marker=marker,
                facecolors="none" if marker == "o" else color,
                edgecolors=color if marker == "o" else "none",
                linewidths=1.6,
                zorder=5,
            )

    strength = [info.agreement if info is not None else -1.0 for info in support]
    strongest = sorted(accepted, key=lambda k: strength[k], reverse=True)[:MAX_LINES]

    for k in strongest:
        i, j = match.pairs[k]
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
                alpha=0.55,
                linewidth=1,
            )
        )

    values = [info.agreement for info in support if info is not None]
    median = f"{np.median(values):.2f}" if values else "n/a"
    pair_name = f"{first_label[0]}/{first_label[1]} ↔ {second_label[0]}/{second_label[1]}"
    print(
        f"{pair_name}: proposals={len(match.pairs)}, consistent={len(accepted)}, "
        f"inconsistent={len(rejected)}, insufficient={len(unknown)}, "
        f"median local agreement={median}"
    )
    figure.suptitle(f"{pair_name}  |  local agreement={median}  |  {len(accepted)} consistent")
    figure.text(
        0.5,
        0.035,
        "Cyan: supported proposals  |  Pink X: inconsistent neighborhood  |  "
        "Orange ring: insufficient evidence  |  Yellow: annotated spots",
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

        draw_pair(first, second, photos[first], photos[second])


if __name__ == "__main__":
    run(Project.open())
