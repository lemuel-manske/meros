"""
Audit spot correspondences without anatomical landmarks.

Only manually annotated spots and image patches are inputs. Small triangles
of nearby spots propose alignment hypotheses; the strongest spatial consensus
supplies tentative correspondences. Verified manual links are used strictly
for evaluation, never to generate or select a transformation.

A geometric correspondence is NOT a confirmed identity. In particular,
dense spot patterns can align by chance across different individuals.
The debug view compares the most-pairs hypothesis with quality-ranked
alternatives, including spatial coverage, geometric residual and appearance.
Close each visual comparison to continue to the next pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import CheckButtons
from scipy.spatial.distance import cdist

import torch  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]

from meros.project import Project


COLOR_SPOT = "#ffcc59"
COLOR_MATCH = "#37f5eb"
COLOR_CORRECT = "#36d399"
COLOR_INCORRECT = "#fb5470"
COLOR_PROJECTION = "#b894ff"
COLOR_BASELINE = "#579bff"
COLOR_NEUTRAL = "#9da4af"
COLOR_SEED = "#ee995f"

DEBUG_RECONSTRUCTION = True
RECONSTRUCTION_HULL_MARGIN = 1.5
RECONSTRUCTION_SPOT_RADIUS = 0.18
RECONSTRUCTION_OVERLAY_ALPHA = 0.55

PATCH_SCALE = 2.0
PATCH_SIZE = 128

TRIANGLE_NEIGHBORS = 5
TRIANGLE_SHAPE_TOLERANCE = 0.11
TRIANGLE_CANDIDATES = 3

MAX_HYPOTHESES = 1800

MATCH_SPACING_RATIO = 0.70

VISUAL_COST_WEIGHT = 0.20

QUALITY_COVERAGE_WEIGHT = 0.70
QUALITY_APPEARANCE_WEIGHT = 0.50

REFINE_TOP = 5

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
class Alignment:
    pairs: tuple[tuple[int, int], ...]
    errors: tuple[float, ...]
    matrix: np.ndarray
    score: float
    coverage: float
    geometry: float
    appearance: float
    seed: tuple[tuple[int, int, int], tuple[int, int, int]] | None


@dataclass(frozen=True)
class Match:
    selected: Alignment
    most_pairs: Alignment
    alternatives: tuple[Alignment, ...]
    scores: tuple[tuple[int, float], ...]
    hypotheses: int


def spot_spacing(points: np.ndarray) -> float:
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.sort(distances, axis=1)[:, 2]))


def create_encoder():
    from torchvision.models import resnet18, ResNet18_Weights  # pyright: ignore[reportMissingImports]

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder = torch.nn.Sequential(*list(model.children())[:-2])
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    return encoder


def load_photo(project: Project, label: Label, encoder) -> Photo:
    points = np.asarray(project.metadata.read_subject_spots(*label), dtype=np.float64)

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

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).repeat(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    batches = []

    with torch.inference_mode():
        for batch in tensor.split(32):
            features = F.adaptive_avg_pool2d(encoder(batch), 1).flatten(1)
            batches.append(F.normalize(features, dim=1).numpy())

    return Photo(points, np.concatenate(batches), image, spacing)


def local_triangles(points: np.ndarray) -> list[Triangle]:
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
    return cv.perspectiveTransform(points.reshape(1, -1, 2), matrix).reshape(-1, 2)


def assign_matches(
    source: Photo,
    target: Photo,
    matrix: np.ndarray,
    appearance: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
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


def alignment_quality(
    source: Photo,
    target: Photo,
    matrix: np.ndarray,
    appearance: np.ndarray,
    seed: tuple[tuple[int, int, int], tuple[int, int, int]] | None,
) -> Alignment:
    pairs, errors = assign_matches(source, target, matrix, appearance)

    if not pairs:
        return Alignment(pairs, errors, matrix, 0.0, 0.0, 0.0, 0.0, seed)

    first, second = zip(*pairs, strict=True)
    indices = np.asarray(first, dtype=int)
    matches = np.asarray(second, dtype=int)
    radius = MATCH_SPACING_RATIO * target.spacing
    normalized = np.asarray(errors) / radius
    geometry = float(np.mean(np.clip(1.0 - normalized**2, 0.0, 1.0)))
    visual = float(np.mean(np.clip(1.0 - appearance[indices, matches], 0.0, 1.0)))

    def fraction(points: np.ndarray, matched: np.ndarray) -> float:
        whole = float(cv.contourArea(cv.convexHull(points.astype(np.float32))))

        if len(matched) < 3 or whole <= 0:
            return 0.0

        area = float(cv.contourArea(cv.convexHull(matched.astype(np.float32))))
        return float(np.clip(area / whole, 0.0, 1.0))

    source_coverage = fraction(source.points, source.points[indices])
    target_coverage = fraction(target.points, target.points[matches])
    coverage = float(np.sqrt(source_coverage * target_coverage))
    support = len(pairs) / np.sqrt(len(source.points) * len(target.points))
    score = (
        support
        * geometry
        * (1.0 - QUALITY_APPEARANCE_WEIGHT + QUALITY_APPEARANCE_WEIGHT * visual)
        * (1.0 - QUALITY_COVERAGE_WEIGHT + QUALITY_COVERAGE_WEIGHT * coverage)
    )
    return Alignment(pairs, errors, matrix, float(score), coverage, geometry, visual, seed)


def match_spots(source: Photo, target: Photo) -> Match:
    first = local_triangles(source.points)
    second = local_triangles(target.points)
    right_shapes = np.array([triangle.shape for triangle in second]).reshape(-1, 2)
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
    candidates: list[Alignment] = []

    for _, left, right in hypotheses:
        matrix = similarity(
            source.points[list(left.indices)],
            target.points[list(right.indices)],
        )
        candidates.append(
            alignment_quality(
                source,
                target,
                matrix,
                appearance,
                (left.indices, right.indices),
            )
        )

    if not candidates:
        empty = alignment_quality(source, target, np.eye(3), appearance, None)
        return Match(empty, empty, (empty,), (), 0)

    most_pairs = max(
        candidates,
        key=lambda item: (len(item.pairs), -float(np.mean(item.errors))),
    )
    initial_scores = tuple((len(item.pairs), item.score) for item in candidates)
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)

    for candidate in ranked[:REFINE_TOP]:
        if len(candidate.pairs) < 4:
            continue

        a, b = zip(*candidate.pairs, strict=True)
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
            refined = alignment_quality(source, target, matrix, appearance, candidate.seed)

            if refined.score > candidate.score:
                candidates.append(refined)

    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    alternatives = []

    for candidate in ranked:
        links = set(candidate.pairs)

        if all(
            len(links & set(previous.pairs)) / max(len(links | set(previous.pairs)), 1) < 0.8
            for previous in alternatives
        ):
            alternatives.append(candidate)

    return Match(ranked[0], most_pairs, tuple(alternatives), initial_scores, len(hypotheses))


def verified_pairs(
    project: Project,
    first: Label,
    second: Label,
    source: Photo,
    target: Photo,
) -> dict[int, int]:
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


def report_match(name: str, alignment: Alignment, truth: dict[int, int]) -> None:
    predicted = dict(alignment.pairs)
    correct = sum(predicted.get(i) == j for i, j in truth.items())
    wrong = sum(i in truth and predicted[i] != truth[i] for i in predicted)
    unknown = len(predicted) - correct - wrong
    print(
        f"  {name}: verified recovered={correct}/{len(truth)}, "
        f"wrong-on-reviewed-source={wrong}, "
        f"unverified-proposals={unknown}"
    )


def draw_debug(
    first_label: Label,
    second_label: Label,
    source: Photo,
    target: Photo,
    result: Match,
    truth: dict[int, int],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    count_axis = axes[0, 0]

    if result.scores:
        counts, scores = zip(*result.scores, strict=True)
        count_axis.scatter(counts, scores, s=7, c=COLOR_NEUTRAL, alpha=0.35)

    count_axis.scatter(
        len(result.most_pairs.pairs),
        result.most_pairs.score,
        s=110,
        color=COLOR_BASELINE,
        marker="x",
        label="Most pairs",
        zorder=5,
    )
    count_axis.scatter(
        len(result.selected.pairs),
        result.selected.score,
        s=90,
        color=COLOR_CORRECT,
        marker="o",
        label="Selected",
        zorder=5,
    )
    count_axis.set_xlabel("Number of assigned pairs")
    count_axis.set_ylabel("Quality: support × residual × appearance × coverage")
    count_axis.legend(loc="best")

    views = [("Quality winner", result.selected), ("Most pairs", result.most_pairs)]

    for candidate in result.alternatives:
        if len(views) >= 3:
            break

        if all(set(candidate.pairs) != set(view.pairs) for _, view in views):
            views.append(("Alternative", candidate))

    for axis, (label, candidate) in zip(axes.flat[1:], views, strict=False):
        axis.imshow(cv.cvtColor(target.image, cv.COLOR_BGR2RGB))
        projected = project(source.points, candidate.matrix)
        axis.scatter(
            target.points[:, 0],
            target.points[:, 1],
            s=10,
            c=COLOR_SPOT,
            alpha=0.9,
        )
        axis.scatter(
            projected[:, 0],
            projected[:, 1],
            s=10,
            c=COLOR_PROJECTION,
            alpha=0.6,
        )

        strongest = sorted(
            zip(candidate.pairs, candidate.errors, strict=True),
            key=lambda pair: pair[1],
        )[:MAX_LINES]

        for (i, j), _ in strongest:
            a, b = projected[i], target.points[j]
            color = COLOR_MATCH

            if i in truth:
                color = COLOR_CORRECT if truth[i] == j else COLOR_INCORRECT
            elif first_label[0] != second_label[0]:
                color = COLOR_INCORRECT

            axis.plot((a[0], b[0]), (a[1], b[1]), c=color, lw=0.8, alpha=0.65)

        if candidate.seed is not None:
            _, indices = candidate.seed
            polygon = target.points[list(indices) + [indices[0]]]
            axis.plot(polygon[:, 0], polygon[:, 1], c=COLOR_SEED, lw=1.8)

        axis.set_xlim(0, target.image.shape[1])
        axis.set_ylim(target.image.shape[0], 0)
        verified = sum(truth.get(i) == j for i, j in candidate.pairs if i in truth)
        wrong = sum(truth.get(i) != j for i, j in candidate.pairs if i in truth)
        axis.set_title(
            f"{label}: score={candidate.score:.3f} | pairs={len(candidate.pairs)} | "
            f"geom={candidate.geometry:.2f} | visual={candidate.appearance:.2f} | "
            f"coverage={candidate.coverage:.2f} | reviewed={verified} ok, {wrong} wrong",
            fontsize=9,
        )
        axis.set_axis_off()

    for axis in list(axes.flat)[1 + len(views) :]:
        axis.set_axis_off()

    figure.suptitle(
        f"DEBUG {first_label[0]}/{first_label[1]} ↔ "
        f"{second_label[0]}/{second_label[1]} — alternatives"
    )
    figure.text(
        0.5,
        0.02,
        "Yellow: target spots | Purple: projected source spots | "
        "Cyan: unverified links | Green: correct | Red: incorrect | Orange: seed triangle",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    plt.show(block=True)
    plt.close(figure)


def draw_reconstruction(
    first_label: Label,
    second_label: Label,
    source: Photo,
    target: Photo,
    alignment: Alignment,
) -> None:
    height, width = source.image.shape[:2]
    margin = max(8, round(RECONSTRUCTION_HULL_MARGIN * source.spacing))
    radius = max(2, round(RECONSTRUCTION_SPOT_RADIUS * source.spacing))

    hull = cv.convexHull(np.rint(source.points).astype(np.int32))
    region = np.zeros((height, width), dtype=np.uint8)
    cv.fillConvexPoly(region, hull, 255)
    cv.polylines(region, [hull], True, 255, thickness=2 * margin)

    binary = region.copy()

    for x, y in source.points:
        cv.circle(binary, (round(x), round(y)), radius, 0, thickness=-1)

    target_height, target_width = target.image.shape[:2]
    warped_mask = cv.warpPerspective(
        binary,
        alignment.matrix,
        (target_width, target_height),
        flags=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_region = cv.warpPerspective(
        region,
        alignment.matrix,
        (target_width, target_height),
        flags=cv.INTER_NEAREST,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    axes[0, 0].imshow(cv.cvtColor(source.image, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original source: {first_label[0]}/{first_label[1]}")

    axes[0, 1].imshow(binary, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0, 1].set_title("Source binary mask (before transform)")

    axes[1, 0].imshow(warped_mask, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1, 0].set_title("Source binary mask after selected transform")

    axes[1, 1].imshow(cv.cvtColor(target.image, cv.COLOR_BGR2RGB))
    overlay_artist = axes[1, 1].imshow(
        warped_mask,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        alpha=RECONSTRUCTION_OVERLAY_ALPHA * (warped_region > 0),
    )
    axes[1, 1].set_title(f"Target: {second_label[0]}/{second_label[1]} + mask")

    for axis in axes.flat:
        axis.set_axis_off()

    check_axis = figure.add_axes((0.64, 0.025, 0.30, 0.045))
    check = CheckButtons(check_axis, ["Show mask on target"], [True])

    def toggle_overlay(_label: str | None) -> None:
        overlay_artist.set_visible(not overlay_artist.get_visible())
        figure.canvas.draw_idle()

    check.on_clicked(toggle_overlay)

    figure.suptitle(
        f"Selected transform ({len(alignment.pairs)} tentative pairs) | "
        "white = approximate region, black holes = source spots"
    )
    figure.text(
        0.04,
        0.035,
        "Binary masks use annotated spot centers and an expanded hull, not fish segmentation.",
        ha="left",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.11, hspace=0.12)
    plt.show(block=True)
    plt.close(figure)


def draw_pair(
    project: Project,
    first_label: Label,
    second_label: Label,
    source: Photo,
    target: Photo,
) -> None:
    result = match_spots(source, target)
    chosen = result.selected
    truth = verified_pairs(project, first_label, second_label, source, target)
    figure, axes = plt.subplots(1, 2, figsize=(17, 8))

    for axis, label, photo in zip(axes, (first_label, second_label), (source, target), strict=True):
        axis.imshow(cv.cvtColor(photo.image, cv.COLOR_BGR2RGB))
        axis.scatter(photo.points[:, 0], photo.points[:, 1], s=14, c=COLOR_SPOT)
        axis.set_title(f"{label[0]}/{label[1]}")
        axis.set_axis_off()

    ordered = np.argsort(chosen.errors)[:MAX_LINES]

    for index in ordered:
        i, j = chosen.pairs[index]
        first_point, second_point = source.points[i], target.points[j]
        color = COLOR_MATCH

        if i in truth:
            color = COLOR_CORRECT if truth[i] == j else COLOR_INCORRECT
        elif first_label[0] != second_label[0]:
            color = COLOR_INCORRECT

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
    residual = f"{np.median(chosen.errors):.1f}px" if chosen.errors else "n/a"
    print(
        f"{name}: pairs={len(chosen.pairs)}, score={chosen.score:.3f}, "
        f"coverage={chosen.coverage:.2f}, geometry={chosen.geometry:.2f}, "
        f"appearance={chosen.appearance:.2f}, residual={residual}, "
        f"count-only={len(result.most_pairs.pairs)} "
        f"(score={result.most_pairs.score:.3f}), hypotheses={result.hypotheses}"
    )

    if truth:
        report_match("most-pairs", result.most_pairs, truth)
        report_match("quality", chosen, truth)

    if first_label[0] != second_label[0]:
        print("  Different biological individuals: links are false identity evidence.")

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

    draw_debug(first_label, second_label, source, target, result, truth)

    if DEBUG_RECONSTRUCTION:
        draw_reconstruction(first_label, second_label, source, target, chosen)


def get_pairs(project: Project) -> list[tuple[Label, Label]]:
    labels = sorted(
        (subject.subject_id, 0)  # pick just the 1st image of each subject
        for subject in project.media.list_subjects()
    )

    pairs = combinations(labels, 2)

    return list(pairs)


def run(project: Project) -> None:
    photos = {}
    pairs = get_pairs(project)

    encoder = create_encoder()

    for first, second in pairs:
        for label in (first, second):
            if label not in photos:
                photos[label] = load_photo(project, label, encoder)

        draw_pair(project, first, second, photos[first], photos[second])


if __name__ == "__main__":
    run(Project.open())
