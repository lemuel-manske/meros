"""
Evaluate individual grouper reidentification with local spot-neighborhood evidence.

A frozen ResNet18 embeds manually annotated spots; eye/mouth landmarks and
geometric matching provide the original score. A fixed geometric-neighborhood
consistency factor can downweight mismatched skin patterns. The baseline stays
available for comparison, and optional full-photo plots support human review.

Edit MODE below to switch between ranking evaluation and visual inspection.
Neither method produces calibrated identity probabilities or automatic decisions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt

from meros.project import Project


PATCH_SCALE = 0.22
PATCH_SIZE = 128
MAX_VISUAL_DISTANCE = 0.35
GEOMETRIC_RADIUS_RATIO = 0.12
GEOMETRIC_WEIGHT = 0.35
RANSAC_RADIUS_RATIO = 0.06
MIN_SPOTS = 4
TOP_K = 3
MODE = "evaluate"  # "evaluate" or "visualize"
NEIGHBOR_RADIUS = 0.40
NEIGHBOR_TOLERANCE = 0.045
ANNOTATED_BORDER = 0.03
MIN_NEIGHBORS = 3
MIN_EVALUABLE_ANCHORS = 3

# Only used in visualization mode. No training or saved files.
VISUAL_PAIRS = (
    (("rafael", 3), ("rogerio", 1)),
    (("rafael", 3), ("rafael", 2)),
    (("rogerio", 1), ("rogerio", 4)),
    (("athila", 0), ("rogerio", 1)),
)
MAX_CONNECTIONS = 20
MIN_SHARED_SPOTS = 5
MIN_ANNOTATED_OVERLAP = 0.20

Label = tuple[str, int]


@dataclass(frozen=True)
class SpotMatch:
    matrix: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    errors: tuple[float, ...]


def initial_similarity(
    source_eye: tuple[float, float],
    source_mouth: tuple[float, float],
    target_eye: tuple[float, float],
    target_mouth: tuple[float, float],
) -> np.ndarray:
    source_origin = np.asarray(source_eye, dtype=float)
    target_origin = np.asarray(target_eye, dtype=float)
    source_axis = np.asarray(source_mouth, dtype=float) - source_origin
    target_axis = np.asarray(target_mouth, dtype=float) - target_origin

    source_length = np.linalg.norm(source_axis)
    target_length = np.linalg.norm(target_axis)

    if source_length == 0 or target_length == 0:
        raise ValueError("Eye and mouth landmarks must be distinct")

    cosine = np.dot(source_axis, target_axis) / (source_length * target_length)
    sine = np.linalg.det(np.stack((source_axis, target_axis))) / (
        source_length * target_length
    )
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    linear = rotation * (target_length / source_length)

    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = target_origin - linear @ source_origin

    return matrix


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(1, -1, 2), matrix
    ).reshape(-1, 2)


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
    import torch  # pyright: ignore[reportMissingImports]
    from torchvision.models import ResNet18_Weights, resnet18  # pyright: ignore[reportMissingImports]

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
    import torch  # pyright: ignore[reportMissingImports]
    import torch.nn.functional as functional  # pyright: ignore[reportMissingImports]

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


@dataclass(frozen=True)
class Candidate:
    subject_id: str
    image_id: int
    score: float


def pair_score(
    source: SpotPhoto,
    target: SpotPhoto,
    match: SpotMatch | None = None,
) -> float:
    """
    Score a pair by matched spot coverage and visual-geometric quality.
    """
    result = match if match is not None else geometric_visual_match(source, target)

    if not result.pairs:
        return 0.0

    radius = GEOMETRIC_RADIUS_RATIO * target.scale
    coverage = len(result.pairs) / min(len(source.points), len(target.points))
    qualities = []

    for (i, j), error in zip(result.pairs, result.errors, strict=True):
        similarity = float(np.dot(source.descriptors[i], target.descriptors[j]))
        visual_quality = max(0.0, 2.0 * similarity - 1.0)
        geometric_quality = max(0.0, 1.0 - error / radius)
        qualities.append(visual_quality * geometric_quality)

    return float(coverage * np.mean(qualities))


def rank_individuals(
    query_label: tuple[str, int],
    query: SpotPhoto,
    gallery: dict[tuple[str, int], SpotPhoto],
    score_cache: dict[tuple[tuple[str, int], tuple[str, int]], float],
    *,
    excluded_subject: str | None = None,
) -> list[Candidate]:
    """
    Rank identities by their best reference, optionally hiding one identity.
    """
    best_by_subject: dict[str, Candidate] = {}

    for (subject_id, image_id), reference in gallery.items():
        if (subject_id, image_id) == query_label or subject_id == excluded_subject:
            continue

        key = (query_label, (subject_id, image_id))

        if key not in score_cache:
            score_cache[key] = pair_score(query, reference)

        score = score_cache[key]
        previous = best_by_subject.get(subject_id)

        if previous is None or score > previous.score:
            best_by_subject[subject_id] = Candidate(subject_id, image_id, score)

    return sorted(
        best_by_subject.values(),
        key=lambda candidate: (-candidate.score, candidate.subject_id),
    )



@dataclass(frozen=True)
class NeighborhoodResult:
    anchors: int
    matched_spots: int
    geometry: float
    local_count_disagreement: float


def maximum_local_matches(
    distance: np.ndarray,
    allowed: np.ndarray,
) -> int:
    """
    Measure one-to-one local support with a minimum-cost assignment.

    Pairs outside the geometric tolerance cost more than unmatched options.
    Keep this assignment unchanged to preserve EXP-004N/O comparability.
    """
    n_source, n_target = distance.shape
    size = n_source + n_target
    cost = np.full((size, size), 3.0, dtype=float)
    cost[:n_source, :n_target] = np.where(
        allowed, np.minimum(distance, 0.99), 3.0
    )
    cost[np.arange(n_source), n_target + np.arange(n_source)] = 0.5
    cost[n_source + np.arange(n_target), np.arange(n_target)] = 0.5
    cost[n_source:, n_target:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    return int(
        sum(
            row < n_source and col < n_target and allowed[row, col]
            for row, col in zip(rows, cols, strict=True)
        )
    )


def inspect_neighborhood(
    source: SpotPhoto,
    target: SpotPhoto,
    match: SpotMatch,
) -> NeighborhoodResult | None:
    """
    Check local constellations surrounding automatically matched anchor spots.

    For each anchor, compare all annotated neighbors inside both shared
    footprints. The anchor position is subtracted, so disagreement is local,
    not a consequence of differences in image translation or crop placement.
    """
    if len(match.pairs) < MIN_EVALUABLE_ANCHORS:
        return None

    projected = project_points(source.points, match.matrix)
    target_points = target.points
    source_hull = cv.convexHull(projected.astype(np.float32))
    target_hull = cv.convexHull(target_points.astype(np.float32))
    area, intersection = cv.intersectConvexConvex(source_hull, target_hull)

    if intersection is None or area <= 0:
        return None

    border = ANNOTATED_BORDER * target.scale

    def comparable(points: np.ndarray) -> set[int]:
        return {
            index
            for index, point in enumerate(points)
            if cv.pointPolygonTest(
                intersection, (float(point[0]), float(point[1])), True
            ) >= border
        }

    shared_source = comparable(projected)
    shared_target = comparable(target_points)
    radius = NEIGHBOR_RADIUS * target.scale
    tolerance = NEIGHBOR_TOLERANCE * target.scale
    geometric = []
    count_disagreement = []

    for source_id, target_id in match.pairs:
        if source_id not in shared_source or target_id not in shared_target:
            continue

        first_ids = [
            i for i in sorted(shared_source)
            if i != source_id and np.linalg.norm(
                projected[i] - projected[source_id]
            ) <= radius
        ]
        second_ids = [
            j for j in sorted(shared_target)
            if j != target_id and np.linalg.norm(
                target_points[j] - target_points[target_id]
            ) <= radius
        ]

        if min(len(first_ids), len(second_ids)) < MIN_NEIGHBORS:
            continue

        first = projected[first_ids] - projected[source_id]
        second = target_points[second_ids] - target_points[target_id]
        distances = cdist(first, second)
        allowed_geometry = distances <= tolerance
        denominator = len(first_ids) + len(second_ids)

        geometric.append(
            2 * maximum_local_matches(distances / tolerance, allowed_geometry)
            / denominator
        )
        count_disagreement.append(
            abs(len(first_ids) - len(second_ids)) / max(len(first_ids), len(second_ids))
        )

    if len(geometric) < MIN_EVALUABLE_ANCHORS:
        return None

    return NeighborhoodResult(
        anchors=len(geometric),
        matched_spots=len(match.pairs),
        geometry=float(np.median(geometric)),
        local_count_disagreement=float(np.median(count_disagreement)),
    )


@dataclass(frozen=True)
class PairEvidence:
    original: float
    neighborhood: NeighborhoodResult | None

    @property
    def combined(self) -> float:
        """
        Preserve the baseline score when no neighborhood can be compared.
        """
        if self.neighborhood is None:
            return self.original

        return self.original * self.neighborhood.geometry


def correct_rank(ranking: list[Candidate], subject_id: str) -> int | None:
    """
    Worst-case true identity rank among ties; zero scores mean no evidence.
    """
    genuine = next((candidate for candidate in ranking if candidate.subject_id == subject_id), None)

    if genuine is None or genuine.score <= 0:
        return None

    return sum(candidate.score >= genuine.score - 1e-9 for candidate in ranking)


def describe_pair(pair: tuple[Label, Label], item: PairEvidence) -> str:
    """
    Display one pair's original score and comparable local geometry.
    """
    first, second = pair
    heading = f"{first[0]}/{first[1]} -> {second[0]}/{second[1]}"

    if item.neighborhood is None:
        return f"{heading} | score={item.original:.3f} | neighborhood=n/a"

    result = item.neighborhood
    return (
        f"{heading} | score={item.original:.3f} | "
        f"local={result.geometry:.3f} ({result.anchors}/{result.matched_spots} anchors) "
        f"| neighbor count difference={result.local_count_disagreement:.3f}"
    )



@dataclass(frozen=True)
class SharedEvidence:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    source_unsupported: tuple[int, ...]
    target_unsupported: tuple[int, ...]
    annotated_overlap: float

    @property
    def density_ratio(self) -> float:
        counts = (len(self.source_indices), len(self.target_indices))
        return max(counts) / min(counts)


def shared_evidence(
    source: SpotPhoto,
    target: SpotPhoto,
    matrix: np.ndarray,
) -> SharedEvidence | None:
    """
    Estimate spot support only within intersecting annotation convex hulls.

    This proxy does not establish identical skin visibility or exhaustive
    annotations. The fitted geometric transformation may also be overfit.
    """
    projected = project_points(source.points, matrix)
    source_hull = cv.convexHull(projected.astype(np.float32))
    target_hull = cv.convexHull(target.points.astype(np.float32))
    source_area = cv.contourArea(source_hull)
    target_area = cv.contourArea(target_hull)

    if min(source_area, target_area) <= 0:
        return None

    area, intersection = cv.intersectConvexConvex(source_hull, target_hull)

    if intersection is None or area <= 0:
        return None

    overlap = float(area / min(source_area, target_area))

    if overlap < MIN_ANNOTATED_OVERLAP:
        return None

    border = ANNOTATED_BORDER * target.scale

    def interior(points: np.ndarray) -> tuple[int, ...]:
        return tuple(
            i
            for i, point in enumerate(points)
            if cv.pointPolygonTest(
                intersection, (float(point[0]), float(point[1])), True
            ) >= border
        )

    source_ids = interior(projected)
    target_ids = interior(target.points)

    if min(len(source_ids), len(target_ids)) < MIN_SHARED_SPOTS:
        return None

    distances = cdist(projected[list(source_ids)], target.points[list(target_ids)])
    radius = GEOMETRIC_RADIUS_RATIO * target.scale
    source_unsupported = tuple(
        source_ids[i] for i in np.where(np.min(distances, axis=1) > radius)[0]
    )
    target_unsupported = tuple(
        target_ids[j] for j in np.where(np.min(distances, axis=0) > radius)[0]
    )

    return SharedEvidence(
        source_ids, target_ids, source_unsupported, target_unsupported, overlap
    )


def draw_spots(
    axis,
    photo: SpotPhoto,
    matched: set[int],
    unsupported: tuple[int, ...],
) -> None:
    """
    Distinguish matched, unmatched, and potentially unsupported annotations.
    """
    unmatched = [i for i in range(len(photo.points)) if i not in matched]

    if unmatched:
        points = photo.points[unmatched]
        axis.scatter(
            points[:, 0], points[:, 1], s=28, c="#ffcf54",
            edgecolors="black", linewidths=0.35, alpha=0.8, zorder=3,
        )

    if matched:
        points = photo.points[sorted(matched)]
        axis.scatter(
            points[:, 0], points[:, 1], s=30, c="#37f5eb",
            edgecolors="black", linewidths=0.45, zorder=4,
        )

    if unsupported:
        points = photo.points[list(unsupported)]
        axis.scatter(
            points[:, 0], points[:, 1], s=75, marker="x", c="#ff416c",
            linewidths=1.5, zorder=5,
        )

    axis.scatter(
        [photo.eye[0]], [photo.eye[1]], marker="+", s=110,
        c="white", linewidths=1.5, zorder=6,
    )
    axis.scatter(
        [photo.mouth[0]], [photo.mouth[1]], marker="s", s=45,
        facecolors="none", edgecolors="white", linewidths=1.5, zorder=6,
    )


def show_pair(
    first_label: tuple[str, int],
    second_label: tuple[str, int],
    first_image: np.ndarray,
    second_image: np.ndarray,
    source: SpotPhoto,
    target: SpotPhoto,
    match: SpotMatch,
    score: float,
    shared: SharedEvidence | None,
) -> None:
    """
    Show full-size photo panels, matched links, and potential absent evidence.
    """
    figure, axes = plt.subplots(1, 2, figsize=(18, 9))
    first_axis, second_axis = axes

    for axis, label, image in zip(
        axes, (first_label, second_label), (first_image, second_image), strict=True
    ):
        axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axis.set_title(f"{label[0]}/{label[1]} — {image.shape[1]}x{image.shape[0]}")
        axis.set_axis_off()

    source_matched = {i for i, _ in match.pairs}
    target_matched = {j for _, j in match.pairs}

    draw_spots(
        first_axis, source, source_matched,
        shared.source_unsupported if shared is not None else (),
    )
    draw_spots(
        second_axis, target, target_matched,
        shared.target_unsupported if shared is not None else (),
    )

    # Show a readable subset; draw ALL matched points on their photos above.
    pair_indices = sorted(
        range(len(match.pairs)), key=lambda index: match.errors[index]
    )[:MAX_CONNECTIONS]

    for line_number, pair_index in enumerate(pair_indices, start=1):
        i, j = match.pairs[pair_index]
        first_point = source.points[i]
        second_point = target.points[j]
        connection = ConnectionPatch(
            xyA=(float(first_point[0]), float(first_point[1])),
            xyB=(float(second_point[0]), float(second_point[1])),
            coordsA="data",
            coordsB="data",
            axesA=first_axis,
            axesB=second_axis,
            color="#37f5eb",
            alpha=0.55,
            linewidth=0.9,
            zorder=2,
        )
        figure.add_artist(connection)

        for axis, point in ((first_axis, first_point), (second_axis, second_point)):
            axis.annotate(
                str(line_number), (float(point[0]), float(point[1])),
                xytext=(4, 4), textcoords="offset points",
                color="white", fontsize=7, fontweight="bold", zorder=7,
            )

    if shared is None:
        region = "Shared annotation region: not comparable"
    else:
        n = len(shared.source_indices) + len(shared.target_indices)
        unsupported = len(shared.source_unsupported) + len(shared.target_unsupported)
        region = (
            f"Annotated overlap={shared.annotated_overlap:.2f}; "
            f"spots={len(shared.source_indices)}+{len(shared.target_indices)}; "
            f"density ratio={shared.density_ratio:.2f}; "
            f"unsupported={unsupported}/{n} ({unsupported / n:.1%})"
        )

    figure.suptitle(
        f"{first_label[0]}/{first_label[1]} → {second_label[0]}/{second_label[1]}"
        f" | original score={score:.3f} | matches={len(match.pairs)}"
    )
    figure.text(
        0.5, 0.055,
        "Cyan: proposed matches (up to 20 numbered links)  |  "
        "Yellow: unpaired annotation  |  "
        "Red X: no nearby spot in comparable annotated region  |  "
        "White +/square: eye/mouth",
        ha="center", va="center", fontsize=10,
    )
    figure.text(
        0.5, 0.025, region + "  |  Red X is NOT proof of a biological difference",
        ha="center", va="center", fontsize=9,
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12, wspace=0.08)

    plt.show(block=True)
    plt.close(figure)



def load_gallery(project: Project, only: set[Label] | None = None) -> dict[Label, SpotPhoto]:
    """
    Load eligible photos once using the Project media/metadata stores.
    """
    encoder = create_encoder()
    gallery: dict[Label, SpotPhoto] = {}

    for subject in project.media.list_subjects():
        for image in subject.images:
            label = (subject.subject_id, image.image_id)

            if only is not None and label not in only:
                continue

            if len(project.metadata.read_subject_spots(*label)) < MIN_SPOTS:
                continue

            photo = load_photo(project, label, encoder)

            if photo is not None:
                gallery[label] = photo

    return gallery


def evaluate(project: Project) -> None:
    """
    Compare baseline and fixed neighborhood scores on all ordered image pairs.
    """
    gallery = load_gallery(project)
    labels = sorted(gallery)
    counts = Counter(subject for subject, _ in labels)
    evidence: dict[tuple[Label, Label], PairEvidence] = {}

    for first in labels:
        for second in labels:
            if first == second:
                continue

            source, target = gallery[first], gallery[second]
            match = geometric_visual_match(source, target)
            original = pair_score(source, target, match)
            local = inspect_neighborhood(source, target, match) if original > 0 else None
            evidence[(first, second)] = PairEvidence(original, local)

    baseline = {pair: item.original for pair, item in evidence.items()}
    combined = {pair: item.combined for pair, item in evidence.items()}

    print(f"Gallery: {len(labels)} photos, {len(counts)} identities")
    print("Methods: unchanged original score / original * local geometry")
    print("Unmeasurable neighborhood: neutral factor (original score preserved).")
    print("Scores are heuristics, not calibrated identity probabilities.")

    for same in (True, False):
        rows = [
            item for (first, second), item in evidence.items()
            if (first[0] == second[0]) == same
        ]
        valid = [item.neighborhood for item in rows if item.neighborhood is not None]
        median = (
            f"{np.median([item.geometry for item in valid]):.3f}"
            if valid else "n/a"
        )
        name = "same" if same else "different"
        print(
            f"{name} ordered pairs: {len(rows)}, comparable={len(valid)}, "
            f"missing={len(rows) - len(valid)}, median local geometry={median}"
        )

    known = [label for label in labels if counts[label[0]] > 1]
    stats: dict[str, dict[Label, tuple[int | None, float, bool]]] = {
        "baseline": {}, "neighborhood": {},
    }

    print("\nKnown queries: best candidate, best impostor, true-vs-impostor margin")

    for label in known:
        for name, scores in (("baseline", baseline), ("neighborhood", combined)):
            ranking = rank_individuals(label, gallery[label], gallery, scores)
            genuine = next(item for item in ranking if item.subject_id == label[0])
            wrong = next(item for item in ranking if item.subject_id != label[0])
            winner = ranking[0]
            rank = correct_rank(ranking, label[0])
            margin = genuine.score - wrong.score
            missing = evidence[(label, (winner.subject_id, winner.image_id))].neighborhood
            stats[name][label] = (rank, margin, missing is None)
            flag = " [no-neighborhood]" if missing is None else ""

            print(
                f"{label[0]}/{label[1]} | {name} | rank={rank or 'n/a'} | "
                f"{winner.subject_id}/{winner.image_id}={winner.score:.3f}{flag} | "
                f"wrong={wrong.subject_id}/{wrong.image_id}={wrong.score:.3f} "
                f"| margin={margin:+.3f}"
            )

    print("\nMethod | Rank-1 | Rank-3 | median margin | top without neighborhood")

    for name, rows in stats.items():
        values = list(rows.values())
        top1 = sum(rank == 1 for rank, _, _ in values)
        topk = sum(rank is not None and rank <= TOP_K for rank, _, _ in values)
        margins = [margin for _, margin, _ in values]
        missing = sum(flag for _, _, flag in values)
        print(
            f"{name} | {top1}/{len(values)} | {topk}/{len(values)} | "
            f"{np.median(margins):+.3f} | {missing}"
        )

    before = stats["baseline"]
    after = stats["neighborhood"]
    improvements = sum(after[label][1] > before[label][1] + 1e-9 for label in known)
    regressions = sum(after[label][1] < before[label][1] - 1e-9 for label in known)
    print(f"Margins: increased={improvements}, decreased={regressions}")

    print("\nSimulated unknown: query identity absent from the gallery")

    for name, scores in (("baseline", baseline), ("neighborhood", combined)):
        top = [
            (
                label,
                rank_individuals(
                    label, gallery[label], gallery, scores, excluded_subject=label[0]
                )[0],
            )
            for label in labels if len(counts) > 1
        ]
        missing = sum(
            evidence[(label, (candidate.subject_id, candidate.image_id))].neighborhood
            is None
            for label, candidate in top
        )
        print(
            f"{name}: queries={len(top)}, "
            f"median best incorrect score={np.median([c.score for _, c in top]):.3f}, "
            f"top without neighborhood={missing}"
        )

    print(
        "Neither open-set rejection nor generalization to new individuals "
        "is validated by these repeated photos."
    )


def visualize(project: Project) -> None:
    """
    Inspect full images with proposed matches and potentially unsupported spots.
    """
    selected = {label for pair in VISUAL_PAIRS for label in pair}
    gallery = load_gallery(project, selected)

    for first, second in VISUAL_PAIRS:
        if first not in gallery or second not in gallery:
            print(f"Skipping {first} -> {second}: photo or annotation unavailable")
            continue

        source, target = gallery[first], gallery[second]
        match = geometric_visual_match(source, target)
        original = pair_score(source, target, match)
        local = inspect_neighborhood(source, target, match) if original > 0 else None
        shared = shared_evidence(source, target, match.matrix)

        metric = f"{local.geometry:.3f}" if local is not None else "n/a"
        print(
            f"{first[0]}/{first[1]} -> {second[0]}/{second[1]} | "
            f"original={original:.3f} | local={metric} | "
            f"combined={original * local.geometry if local is not None else original:.3f}"
        )
        show_pair(
            first, second,
            project.media.read_subject_image(*first),
            project.media.read_subject_image(*second),
            source, target, match, original, shared,
        )

    print(
        "Unmatched spots and red X are diagnostics, not confirmed "
        "biological differences."
    )


def run(project: Project) -> None:
    """
    Run the single configured experiment entry point.
    """
    if MODE == "evaluate":
        evaluate(project)
    elif MODE == "visualize":
        visualize(project)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    run(Project.open())
