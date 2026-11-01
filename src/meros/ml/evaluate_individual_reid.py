"""
Audit spot correspondences without anatomical landmarks.

Only manually annotated spots and image patches are inputs. Small triangles
of nearby spots propose alignment hypotheses; the strongest spatial consensus
supplies tentative correspondences. Verified manual links are used strictly
for evaluation, never to generate or select a transformation.

A geometric correspondence is NOT a confirmed identity. In particular,
dense spot patterns can align by chance across different individuals.
Close each visual comparison to continue to the next pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

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
    (("rogerio", 0), ("rogerio", 1)),
    (("rogerio", 11), ("rogerio", 12)),
)

MIN_SPOTS = 4
PATCH_SCALE = 2.0
PATCH_SIZE = 128
TRIANGLE_NEIGHBORS = 5
TRIANGLE_SHAPE_TOLERANCE = 0.11
TRIANGLE_CANDIDATES = 3
MAX_HYPOTHESES = 1800
MATCH_SPACING_RATIO = 0.70
VISUAL_COST_WEIGHT = 0.20
TRUTH_SPACING_RATIO = 0.50
MAX_LINES = 20

Label = tuple[str, int]


@dataclass(frozen=True)
class Photo:
    points: np.ndarray
    descriptors: np.ndarray
    image: np.ndarray
    spacing: float


@dataclass(frozen=True)
class Triangle:
    indices: tuple[int, int, int]
    shape: tuple[float, float]
    orientation: int


@dataclass(frozen=True)
class Match:
    pairs: tuple[tuple[int, int], ...]
    errors: tuple[float, ...]
    matrix: np.ndarray
    hypotheses: int


def spot_spacing(points: np.ndarray) -> float:
    """
    Estimate patch and matching scale from spots alone, never anatomy.
    """
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.sort(distances, axis=1)[:, 2]))


def create_encoder():
    """
    Load frozen ImageNet ResNet18 features, as in the earlier experiment.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from torchvision.models import ResNet18_Weights  # pyright: ignore[reportMissingImports]
    from torchvision.models import resnet18  # pyright: ignore[reportMissingImports]

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder = torch.nn.Sequential(*list(model.children())[:-2])
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    return encoder


def load_photo(project: Project, label: Label, encoder) -> Photo:
    """
    Read spots and image through Project. No other annotations are required.
    """
    points = np.asarray(project.metadata.read_subject_spots(*label), dtype=np.float64)

    if len(points) < MIN_SPOTS:
        raise ValueError(f"{label}: fewer than {MIN_SPOTS} annotated spots")

    image = project.media.read_subject_image(*label)
    spacing = spot_spacing(points)
    width = max(16, round(PATCH_SCALE * spacing))
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    equalizer = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    patches = []

    for x, y in points:
        patch = cv.getRectSubPix(grayscale, (width, width), (float(x), float(y)))
        patch = cv.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv.INTER_AREA)
        patches.append(equalizer.apply(patch).astype(np.float32) / 255.0)

    import torch  # pyright: ignore[reportMissingImports]
    import torch.nn.functional as functional  # pyright: ignore[reportMissingImports]

    tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    batches = []

    with torch.inference_mode():
        for batch in tensor.split(32):
            features = functional.adaptive_avg_pool2d(encoder(batch), 1).flatten(1)
            batches.append(functional.normalize(features, dim=1).numpy())

    return Photo(points, np.concatenate(batches), image, spacing)


def local_triangles(points: np.ndarray) -> list[Triangle]:
    """
    Describe local triangles by ratios, invariant to translation and scale.
    """
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    result = []

    for i in range(len(points)):
        nearest = np.argsort(distances[i])[:TRIANGLE_NEIGHBORS]

        for j, k in combinations(nearest, 2):
            first = points[j] - points[i]
            second = points[k] - points[i]
            a = float(np.linalg.norm(first))
            b = float(np.linalg.norm(second))
            cross = float(np.linalg.det(np.stack((first, second))))

            if min(a, b) <= 1e-6 or abs(cross) < 0.15 * a * b:
                continue

            c = float(np.linalg.norm(points[j] - points[k]))
            result.append(
                Triangle(
                    (i, int(j), int(k)),
                    (a / b, c / b),
                    1 if cross > 0 else -1,
                )
            )

    return result


def similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Fit a similarity transform from three candidate corresponding spots.
    """
    p = first.mean(axis=0)
    q = second.mean(axis=0)
    x = first - p
    y = second - q
    denominator = float(np.sum(x * x))
    a = float(np.sum(x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]) / denominator)
    b = float(np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]) / denominator)
    linear = np.array(((a, -b), (b, a)))
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = q - linear @ p
    return matrix


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Project all source spots into the target image coordinate system.
    """
    return cv.perspectiveTransform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)


def assign_matches(
    source: Photo,
    target: Photo,
    matrix: np.ndarray,
    appearance: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    """
    Greedily select spatially close, one-to-one spot hypotheses.
    """
    distances = cdist(project(source.points, matrix), target.points)
    radius = MATCH_SPACING_RATIO * target.spacing
    first, second = np.where(distances <= radius)
    order = np.argsort(
        distances[first, second] / radius + VISUAL_COST_WEIGHT * appearance[first, second],
        kind="stable",
    )
    used_source: set[int] = set()
    used_target: set[int] = set()
    pairs = []
    errors = []

    for position in order:
        i, j = int(first[position]), int(second[position])

        if i in used_source or j in used_target:
            continue

        used_source.add(i)
        used_target.add(j)
        pairs.append((i, j))
        errors.append(float(distances[i, j]))

    return tuple(pairs), tuple(errors)


def match_spots(source: Photo, target: Photo) -> Match:
    """
    Generate initial alignment hypotheses ONLY from local spot triangles.

    The highest-consensus hypothesis is exploratory. Many correspondences
    alone do not establish that both photos show the same individual.
    """
    first = local_triangles(source.points)
    second = local_triangles(target.points)
    right_shapes = np.array([triangle.shape for triangle in second])
    right_orientations = np.array([triangle.orientation for triangle in second])
    hypotheses = []
    appearance = cdist(source.descriptors, target.descriptors, metric="cosine")

    for triangle in first:
        differences = np.linalg.norm(right_shapes - triangle.shape, axis=1)
        differences[right_orientations != triangle.orientation] = np.inf

        for index in np.argsort(differences)[:TRIANGLE_CANDIDATES]:
            if differences[index] <= TRIANGLE_SHAPE_TOLERANCE:
                hypotheses.append((float(differences[index]), triangle, second[index]))

    hypotheses.sort(key=lambda item: item[0])
    hypotheses = hypotheses[:MAX_HYPOTHESES]
    best_pairs: tuple[tuple[int, int], ...] = ()
    best_errors: tuple[float, ...] = ()
    best_matrix = np.eye(3)
    best_quality = (-1, float("-inf"))

    for _, left, right in hypotheses:
        matrix = similarity(
            source.points[list(left.indices)],
            target.points[list(right.indices)],
        )
        pairs, errors = assign_matches(source, target, matrix, appearance)
        quality = (
            len(pairs),
            -float(np.mean(errors)) if errors else float("-inf"),
        )

        if quality > best_quality:
            best_pairs, best_errors, best_matrix, best_quality = (pairs, errors, matrix, quality)

    if len(best_pairs) >= 4:
        a, b = zip(*best_pairs, strict=True)
        cv.setRNGSeed(0)

        try:
            affine, inliers = cv.estimateAffine2D(
                source.points[list(a)],
                target.points[list(b)],
                method=cv.RANSAC,
                ransacReprojThreshold=MATCH_SPACING_RATIO * target.spacing,
            )
        except cv.error:
            affine, inliers = None, None

        if affine is not None and inliers is not None and int(inliers.sum()) >= 4:
            matrix = np.vstack((affine, [0.0, 0.0, 1.0]))
            pairs, errors = assign_matches(source, target, matrix, appearance)

            if len(pairs) >= len(best_pairs):
                best_pairs, best_errors, best_matrix = pairs, errors, matrix

    return Match(best_pairs, best_errors, best_matrix, len(hypotheses))


def verified_pairs(
    project: Project,
    first: Label,
    second: Label,
    source: Photo,
    target: Photo,
) -> dict[int, int]:
    """
    Convert existing human-verified coordinates into spot indices for audit.
    """
    if first[0] != second[0]:
        return {}

    reverse = first[1] > second[1]
    left_id, right_id = sorted((first[1], second[1]))
    records = project.metadata.read_subject_spot_correspondences(first[0], left_id, right_id)
    truth: dict[int, int] = {}
    used_targets: set[int] = set()

    for record in records:
        left = record["second"] if reverse else record["first"]
        right = record["first"] if reverse else record["second"]
        d_source = np.linalg.norm(source.points - left, axis=1)
        d_target = np.linalg.norm(target.points - right, axis=1)
        i, j = int(np.argmin(d_source)), int(np.argmin(d_target))

        if d_source[i] > TRUTH_SPACING_RATIO * source.spacing:
            continue

        if d_target[j] > TRUTH_SPACING_RATIO * target.spacing:
            continue

        if i not in truth and j not in used_targets:
            truth[i] = j
            used_targets.add(j)

    if records:
        print(f"  verified annotations mapped to spots: {len(truth)}/{len(records)}")

    return truth


def report_match(match: Match, truth: dict[int, int]) -> None:
    """
    Evaluate only reviewed source spots; unreviewed predictions stay unknown.
    """
    predicted = dict(match.pairs)
    correct = sum(predicted.get(i) == j for i, j in truth.items())
    wrong = sum(i in truth and predicted[i] != truth[i] for i in predicted)
    unknown = len(predicted) - correct - wrong
    print(
        f"  verified recovered={correct}/{len(truth)}, "
        f"wrong-on-reviewed-source={wrong}, "
        f"unverified-proposals={unknown}"
    )


def draw_pair(
    project: Project,
    first_label: Label,
    second_label: Label,
    source: Photo,
    target: Photo,
) -> None:
    """
    Always visualize geometric hypotheses; never label them confirmed identity.
    """
    match = match_spots(source, target)
    truth = verified_pairs(project, first_label, second_label, source, target)
    figure, axes = plt.subplots(1, 2, figsize=(17, 8))

    for axis, label, photo in zip(axes, (first_label, second_label), (source, target), strict=True):
        axis.imshow(cv.cvtColor(photo.image, cv.COLOR_BGR2RGB))
        axis.scatter(photo.points[:, 0], photo.points[:, 1], s=14, c="#ffcc59")
        axis.set_title(f"{label[0]}/{label[1]}")
        axis.set_axis_off()

    ordered = np.argsort(match.errors)[:MAX_LINES]

    for index in ordered:
        i, j = match.pairs[index]
        first_point, second_point = source.points[i], target.points[j]
        color = "#37f5eb"

        if i in truth:
            color = "#36d399" if truth[i] == j else "#fb5470"
        elif first_label[0] != second_label[0]:
            color = "#fb5470"

        figure.add_artist(
            ConnectionPatch(
                xyA=(float(first_point[0]), float(first_point[1])),
                xyB=(float(second_point[0]), float(second_point[1])),
                coordsA="data",
                coordsB="data",
                axesA=axes[0],
                axesB=axes[1],
                color=color,
                alpha=0.60,
                linewidth=1.2,
            )
        )

    name = f"{first_label[0]}/{first_label[1]} ↔ {second_label[0]}/{second_label[1]}"
    residual = f"{np.median(match.errors):.1f}px" if match.errors else "n/a"
    print(
        f"{name}: proposals={len(match.pairs)}, "
        f"alignment hypotheses={match.hypotheses}, "
        f"median residual={residual}"
    )

    if truth:
        report_match(match, truth)

    if first_label[0] != second_label[0]:
        print("  Different biological individuals: all proposed links require rejection.")

    figure.suptitle(f"{name} — tentative geometric associations")
    figure.text(
        0.5,
        0.035,
        "Green: verified link | Red: known mismatch or different individuals | "
        "Cyan: unverified | Yellow: annotated spots",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10, wspace=0.04)
    plt.show(block=True)
    plt.close(figure)


def run(project: Project) -> None:
    """
    Inspect all selected pairs using only spot and image annotations.
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
