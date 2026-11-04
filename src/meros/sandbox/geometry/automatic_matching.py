"""
Compare label-free geometric spot matching using Shape Context and local triangles.

Both approaches propose affine hypotheses without known spot pairs, fit a
regularized TPS to consensus correspondences, and evaluate held-out source
spots against unused target spots. No eye, mouth, image or pair annotations
are used. Scores are exploratory, not calibrated identity probabilities.

Run: python -m meros.sandbox.geometry.automatic_matching
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import cdist

from meros.project import Project


SOURCE = ("rogerio", 10)
TARGETS = (("rogerio", 12), ("rafael", 3))
METHODS = ("shape_context", "triangles")
SEED = 17
REPEATS = 3
TRAIN_FRACTION = 0.65
HYPOTHESES = 1500
CANDIDATES_PER_SPOT = 4
TRIANGLE_TOLERANCE = 0.12
MATCH_RADIUS_FACTOR = 0.65
MAX_AFFINE_CONDITION = 3.0
MIN_RELATIVE_SCALE = 0.35
MAX_RELATIVE_SCALE = 3.0
TPS_SMOOTHING = 1.0
MAX_TPS_DISPLACEMENT = 1.3
NULL_SHIFTS = 24
SHOW_PLOTS = True


@dataclass(frozen=True)
class MatchResult:
    method: str
    source: np.ndarray
    target: np.ndarray
    projected: np.ndarray
    anchor_source: np.ndarray
    anchor_target: np.ndarray
    held_source: np.ndarray
    held_target: np.ndarray
    evaluated: int
    observed: int
    expected: float
    score: float
    bending: float


def spacing(points: np.ndarray) -> float:
    """
    Measure typical spot separation in native image coordinates.
    """
    distances, _ = cKDTree(points).query(points, k=2)
    return max(float(np.median(distances[:, 1])), 1.0)


def shape_context(points: np.ndarray) -> np.ndarray:
    """
    Compute log-polar local histograms, invariant to discrete angular shifts.
    """
    delta = points[None, :, :] - points[:, None, :]
    radius = np.linalg.norm(delta, axis=2) / spacing(points)
    angle = np.mod(np.arctan2(delta[:, :, 1], delta[:, :, 0]), 2 * np.pi)
    bins = np.array((0.0, 0.8, 1.6, 3.0, 5.5, 10.0))
    result = np.zeros((len(points), len(bins) - 1, 8))

    for index in range(len(points)):
        selected = (radius[index] > 1e-6) & (radius[index] < bins[-1])
        r = np.searchsorted(bins, radius[index, selected], side="right") - 1
        a = np.floor(8 * angle[index, selected] / (2 * np.pi)).astype(int)
        np.add.at(result[index], (r, a), 1)

    total = result.sum(axis=(1, 2), keepdims=True)
    return result / np.maximum(total, 1)


def shape_candidates(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Return best target candidates for every source descriptor.
    """
    left = shape_context(source)
    right = shape_context(target)
    costs = np.full((len(source), len(target)), np.inf)

    for shift in range(8):
        rotated = np.roll(right, shift, axis=2)
        differences = left[:, None] - rotated[None, :]
        totals = left[:, None] + rotated[None, :] + 1e-8
        cost = 0.5 * np.sum(differences**2 / totals, axis=(2, 3))
        costs = np.minimum(costs, cost)

    return np.argsort(costs, axis=1)[:, :CANDIDATES_PER_SPOT]


def local_triangles(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Describe Delaunay triangles using side ratios and canonical vertex order.
    """
    triangles = Delaunay(points).simplices
    signatures = []
    ordered = []

    for indices in triangles:
        vertices = points[indices]
        opposite = np.array(
            (
                np.linalg.norm(vertices[1] - vertices[2]),
                np.linalg.norm(vertices[0] - vertices[2]),
                np.linalg.norm(vertices[0] - vertices[1]),
            )
        )
        order = np.argsort(opposite)
        ordered.append(indices[order])
        signatures.append(np.sort(opposite) / opposite.sum())

    return np.asarray(ordered, dtype=int), np.asarray(signatures)


def hypotheses(
    source: np.ndarray, target: np.ndarray, method: str, rng: np.random.Generator
):
    """
    Yield affine hypotheses from unlabeled geometric correspondences.
    """
    if method == "shape_context":
        candidates = shape_candidates(source, target)

        for _ in range(HYPOTHESES):
            first = rng.choice(len(source), 3, replace=False)
            second = candidates[first, rng.integers(CANDIDATES_PER_SPOT, size=3)]

            if len(set(second)) < 3:
                continue

            yield source[first], target[second]

    elif method == "triangles":
        first, first_signature = local_triangles(source)
        second, second_signature = local_triangles(target)
        close = cdist(first_signature, second_signature) < TRIANGLE_TOLERANCE
        rows, columns = np.where(close)

        if not len(rows):
            return

        selected = rng.permutation(len(rows))[:HYPOTHESES]

        for index in selected:
            yield source[first[rows[index]]], target[second[columns[index]]]

    else:
        raise ValueError(f"Unknown matching method: {method}")


def plausible(matrix: np.ndarray, scale_ratio: float) -> bool:
    """
    Reject flipped, collapsed or extremely distorted 2D geometry.
    """
    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))

    if determinant <= 1e-8:
        return False

    singular = np.linalg.svd(linear, compute_uv=False)
    relative_scale = np.sqrt(determinant) / scale_ratio
    return bool(
        singular[-1] > 1e-6
        and singular[0] / singular[-1] <= MAX_AFFINE_CONDITION
        and MIN_RELATIVE_SCALE <= relative_scale <= MAX_RELATIVE_SCALE
    )


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def unique_links(
    projected: np.ndarray, target: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match spots greedily one-to-one within a fixed geometric gate.
    """
    distances, neighbors = cKDTree(target).query(projected, k=min(4, len(target)))
    distances = np.atleast_2d(distances)
    neighbors = np.atleast_2d(neighbors)

    if distances.shape[0] != len(projected):
        distances = distances.T
        neighbors = neighbors.T

    proposals = []

    for index in range(len(projected)):
        for distance, neighbor in zip(distances[index], neighbors[index], strict=True):
            if distance < radius:
                proposals.append((float(distance), index, int(neighbor)))

    proposals.sort()
    occupied_source: set[int] = set()
    occupied_target: set[int] = set()
    accepted = []

    for distance, first, second in proposals:
        if first in occupied_source or second in occupied_target:
            continue

        accepted.append((first, second, distance))
        occupied_source.add(first)
        occupied_target.add(second)

    if not accepted:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])

    return (
        np.array([item[0] for item in accepted], dtype=int),
        np.array([item[1] for item in accepted], dtype=int),
        np.array([item[2] for item in accepted], dtype=float),
    )


def best_hypothesis(
    source: np.ndarray, target: np.ndarray, method: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the transformation from seed-set consensus, never held-out spots.
    """
    radius = MATCH_RADIUS_FACTOR * spacing(target)
    scale_ratio = spacing(target) / spacing(source)
    best_score = -np.inf
    best = None

    for left, right in hypotheses(source, target, method, rng):
        affine = cv.getAffineTransform(left.astype(np.float32), right.astype(np.float32))

        if not plausible(affine, scale_ratio):
            continue

        mapped = transform(source, affine)
        indices, matches, residual = unique_links(mapped, target, radius)

        if len(indices) < 6:
            continue

        span = np.ptp(target[matches], axis=0) / np.maximum(np.ptp(target, axis=0), 1)
        coverage = float(np.sqrt(np.clip(np.prod(span), 0, 1)))
        consensus = len(indices) + 3 * coverage - float(np.median(residual)) / radius

        if consensus > best_score:
            best_score = consensus
            best = affine

    if best is None:
        raise RuntimeError(f"No plausible {method} hypothesis with six geometric matches")

    for _ in range(2):
        indices, matches, _ = unique_links(transform(source, best), target, radius)
        design = np.column_stack((source[indices], np.ones(len(indices))))
        coefficients = np.linalg.lstsq(design, target[matches], rcond=None)[0]
        refined = coefficients.T

        if plausible(refined, scale_ratio):
            best = refined

    indices, matches, _ = unique_links(transform(source, best), target, radius)
    return best, indices, matches


def fit_tps(
    source: np.ndarray, target: np.ndarray, affine: np.ndarray, first: np.ndarray, second: np.ndarray
):
    """
    Fit a regularized bounded residual warp from unlabeled consensus pairs.
    """
    scale = spacing(target)
    reference = transform(source[first], affine)
    residual = (target[second] - reference) / scale
    tps = RBFInterpolator(
        reference / scale, residual, kernel="thin_plate_spline", smoothing=TPS_SMOOTHING
    )

    def project(points: np.ndarray) -> np.ndarray:
        base = transform(points, affine)
        deformation = tps(base / scale)
        lengths = np.linalg.norm(deformation, axis=1)
        ratios = np.minimum(1, MAX_TPS_DISPLACEMENT / np.maximum(lengths, 1e-8))
        return base + scale * deformation * ratios[:, None]

    bending = float(np.median(np.linalg.norm(residual, axis=1)))
    return project, bending


def heldout_score(
    projected: np.ndarray, target: np.ndarray, occupied: np.ndarray,
    radius: float, rng: np.random.Generator,
) -> tuple[int, int, float, float, np.ndarray, np.ndarray]:
    """
    Verify on unused source spots and targets, correcting for chance coincidence.
    """
    available = np.ones(len(target), dtype=bool)
    available[occupied] = False
    remaining = target[available]
    lower = np.min(target, axis=0)
    upper = np.max(target, axis=0)
    valid = np.all((projected >= lower - radius) & (projected <= upper + radius), axis=1)
    evaluated = projected[valid]

    if not len(evaluated) or not len(remaining):
        return 0, 0, 0.0, 0.0, np.empty((0, 2)), np.empty((0, 2))

    first, second, _ = unique_links(evaluated, remaining, radius)
    observed = len(first)
    random_counts = []
    width = np.maximum(upper - lower, 1)

    for _ in range(NULL_SHIFTS):
        displacement = rng.uniform(0, width)
        shifted = lower + np.mod(remaining - lower + displacement, width)
        other_first, _, _ = unique_links(evaluated, shifted, radius)
        random_counts.append(len(other_first))

    expected = float(np.mean(random_counts))
    excess = max(0, observed - expected) / len(evaluated)
    return len(evaluated), observed, expected, excess, evaluated[first], remaining[second]


def compare(
    source: np.ndarray, target: np.ndarray, method: str, rng: np.random.Generator
) -> MatchResult:
    """
    Fit without pair labels and score spots unavailable during model selection.
    """
    training_count = max(8, round(TRAIN_FRACTION * len(source)))
    training_indices = rng.permutation(len(source))
    training = source[training_indices[:training_count]]
    held = source[training_indices[training_count:]]

    affine, anchors, matches = best_hypothesis(training, target, method, rng)
    project, bending = fit_tps(training, target, affine, anchors, matches)
    transformed = project(source)
    threshold = MATCH_RADIUS_FACTOR * spacing(target)
    evaluated, observed, expected, excess, held_source, held_target = heldout_score(
        project(held), target, matches, threshold, rng
    )
    support = min(1.0, len(anchors) / max(8, 0.35 * len(training)))
    penalty = np.exp(-bending / MAX_TPS_DISPLACEMENT)
    score = float(excess * support * penalty)

    return MatchResult(
        method, source, target, transformed, training[anchors], target[matches],
        held_source, held_target, evaluated, observed, expected, score, bending,
    )


def visualize(results: dict[str, MatchResult], title: str) -> None:
    """
    Plot geometry and matched held-out points for human inspection.
    """
    figure, axes = plt.subplots(1, len(results), figsize=(13, 6), squeeze=False)

    for axis, (method, result) in zip(axes[0], results.items(), strict=True):
        axis.scatter(result.target[:, 0], result.target[:, 1], s=18, color="#ffcc59")
        axis.scatter(
            result.projected[:, 0],
            result.projected[:, 1],
            s=17,
            facecolors="none",
            edgecolors="#33ccdd",
        )

        for left, right in zip(result.held_source, result.held_target, strict=True):
            axis.plot((left[0], right[0]), (left[1], right[1]), color="#ff6666", alpha=0.6)

        axis.set_aspect("equal", adjustable="box")
        axis.invert_yaxis()
        axis.set_title(
            f"{method}: held-out={result.observed}/{result.evaluated}; "
            f"score={result.score:.2f}"
        )

    figure.suptitle(title + " | cyan: mapped source | yellow: target | red: held-out links")
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


def run() -> None:
    project = Project.open()
    source = np.asarray(project.metadata.read_subject_spots(*SOURCE), dtype=float)

    for target_id in TARGETS:
        target = np.asarray(project.metadata.read_subject_spots(*target_id), dtype=float)
        print(f"\n{SOURCE} -> {target_id} | spots {len(source)} -> {len(target)}")
        plots = {}

        for method in METHODS:
            results = []

            for repeat in range(REPEATS):
                rng = np.random.default_rng(SEED + repeat)

                try:
                    result = compare(source, target, method, rng)
                except (RuntimeError, np.linalg.LinAlgError, ValueError) as error:
                    print(f"  {method} repeat {repeat}: {error}")
                    continue

                results.append(result)

            if not results:
                print(f"  {method}: no plausible alignment")
                continue

            scores = np.array([result.score for result in results])
            print(
                f"  {method:14s} score={np.mean(scores):.3f} +/- {np.std(scores):.3f} "
                f"(chance-corrected, not calibrated) | "
                f"first split: anchors={len(results[0].anchor_source)}, "
                f"held-out={results[0].observed}/{results[0].evaluated}, "
                f"chance={results[0].expected:.1f}"
            )
            plots[method] = results[0]

        if SHOW_PLOTS and plots:
            visualize(plots, f"{SOURCE} -> {target_id}")


if __name__ == "__main__":
    run()
