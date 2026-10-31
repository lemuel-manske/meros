"""
Identify individual groupers using frozen visual spot features and geometry.

Each photo is queried against all other eligible photos; the best-scoring
reference represents each identity. The reference image itself is excluded.
Scores are fixed heuristics, not calibrated identity probabilities.
No manually verified spot correspondences are used for identification.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist

from meros.project import Project


PATCH_SCALE = 0.22
PATCH_SIZE = 128
MAX_VISUAL_DISTANCE = 0.35
GEOMETRIC_RADIUS_RATIO = 0.12
GEOMETRIC_WEIGHT = 0.35
RANSAC_RADIUS_RATIO = 0.06
MIN_SPOTS = 4
TOP_K = 3
OPEN_SET_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)


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


def pair_score(source: SpotPhoto, target: SpotPhoto) -> float:
    """
    Score a pair by matched spot coverage and visual-geometric quality.
    """
    result = geometric_visual_match(source, target)

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


def report_open_set(
    gallery: dict[tuple[str, int], SpotPhoto],
    score_cache: dict[tuple[tuple[str, int], tuple[str, int]], float],
    known_results: list[tuple[float, bool]],
) -> None:
    """
    Hide each individual's entire gallery and measure false-match scores.
    """
    unknown_scores = []

    print("\nOpen-set: every query identity is completely absent from its gallery")
    print("held-out query | best incorrect identity | max_score | top2_margin")

    for label, photo in gallery.items():
        ranking = rank_individuals(
            label,
            photo,
            gallery,
            score_cache,
            excluded_subject=label[0],
        )

        if not ranking:
            print(f"{label[0]}/{label[1]} | no other identities available")
            continue

        best = ranking[0]
        margin = (
            f"{best.score - ranking[1].score:.3f}" if len(ranking) >= 2 else "n/a"
        )
        unknown_scores.append(best.score)
        print(
            f"{label[0]}/{label[1]} | {best.subject_id}/{best.image_id} | "
            f"{best.score:.3f} | {margin}"
        )

    if not unknown_scores or not known_results:
        return

    known_scores = [score for score, correct in known_results if correct]

    if known_scores:
        print(
            f"Correct known-match top scores: min={min(known_scores):.3f}, "
            f"median={np.median(known_scores):.3f}, max={max(known_scores):.3f}"
        )

    print(
        f"Unknown top scores: min={min(unknown_scores):.3f}, "
        f"median={np.median(unknown_scores):.3f}, max={max(unknown_scores):.3f}"
    )
    print("\nIllustrative score thresholds (not calibrated)")
    print("threshold | known correctly accepted | unknown falsely accepted")

    for threshold in OPEN_SET_THRESHOLDS:
        true_accepts = sum(
            correct and score >= threshold for score, correct in known_results
        )
        false_accepts = sum(score >= threshold for score in unknown_scores)
        print(
            f"{threshold:.2f} | {true_accepts}/{len(known_results)} "
            f"({true_accepts / len(known_results):.1%}) | "
            f"{false_accepts}/{len(unknown_scores)} "
            f"({false_accepts / len(unknown_scores):.1%})"
        )

    print("Scores are not calibrated; these thresholds are only diagnostics.")
    print(
        "Known queries have a true match in the gallery; unknown queries do not. "
        "Gallery size also varies across held-out identities."
    )


def run(project: Project) -> None:
    """
    Evaluate Rank-1, Rank-k, and reciprocal rank over independent subject photos.
    """
    encoder = create_encoder()
    gallery: dict[tuple[str, int], SpotPhoto] = {}

    for subject in project.media.list_subjects():
        for image in subject.images:
            label = (subject.subject_id, image.image_id)
            points = project.metadata.read_subject_spots(*label)

            if len(points) < MIN_SPOTS:
                continue

            photo = load_photo(project, label, encoder)

            if photo is not None:
                gallery[label] = photo
            else:
                print(f"SKIP {label}: missing eye/mouth or invalid annotations")

    counts = Counter(label[0] for label in gallery)
    queries = [label for label in gallery if counts[label[0]] >= 2]
    score_cache: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}
    known_results: list[tuple[float, bool]] = []

    print(
        f"Gallery: {len(gallery)} images, {len(counts)} identities; "
        f"{len(queries)} leave-one-image-out queries."
    )
    print("Method: visual + geometry (frozen ImageNet ResNet18).")
    print("Scores are heuristics, not probabilities.")
    print("query | prediction | conservative_true_rank | top candidates (score)")

    correct_rank1 = 0
    correct_rankk = 0
    reciprocal_ranks = []
    ties = 0

    for label in queries:
        ranking = rank_individuals(label, gallery[label], gallery, score_cache)
        true_candidate = next(
            (candidate for candidate in ranking if candidate.subject_id == label[0]),
            None,
        )

        if true_candidate is None:
            print(f"{label[0]}/{label[1]} | no reference for true identity")
            continue

        # Worst rank among equal scores; do not reward alphabetical tie-breaking.
        true_rank = sum(
            candidate.score >= true_candidate.score - 1e-9
            for candidate in ranking
        )
        best_score = ranking[0].score
        tied_best = sum(abs(candidate.score - best_score) <= 1e-9 for candidate in ranking)
        is_tied = tied_best > 1
        ties += int(is_tied)
        correct = true_rank == 1 and best_score > 0
        correct_rank1 += int(correct)
        known_results.append((best_score, correct))
        correct_rankk += int(true_rank <= TOP_K and true_candidate.score > 0)
        reciprocal_ranks.append(1.0 / true_rank)

        top = ", ".join(
            f"{candidate.subject_id}/{candidate.image_id}={candidate.score:.3f}"
            for candidate in ranking[:TOP_K]
        )
        suffix = " [TIE]" if is_tied else ""

        print(
            f"{label[0]}/{label[1]} | {ranking[0].subject_id} | "
            f"{true_rank}/{len(ranking)} | {top}{suffix}"
        )

    evaluated = len(reciprocal_ranks)

    if not evaluated:
        print("No eligible queries.")
        return

    print(
        f"Rank-1 (unique winner): {correct_rank1}/{evaluated} "
        f"({correct_rank1 / evaluated:.1%})"
    )
    print(f"Rank-{TOP_K}: {correct_rankk}/{evaluated} ({correct_rankk / evaluated:.1%})")
    print(f"Conservative mean reciprocal rank: {np.mean(reciprocal_ranks):.3f}")
    print(f"Ambiguous best-score ties: {ties}/{evaluated}")

    report_open_set(gallery, score_cache, known_results)


if __name__ == "__main__":
    run(Project.open())
