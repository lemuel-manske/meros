"""Prototype positive-unlabeled spot candidate ranking with leave-one-video-out evaluation."""

from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from scipy.spatial import cKDTree

from meros.project import Project
from meros.sandbox import spot_log, spot_blobs, spot_segmentation
from meros.sandbox.evaluate_spots import evaluate


# Edit these values directly in the sandbox.
POSITIVE_DISTANCE_PX = 5.0
UNLABELED_DISTANCE_PX = 16.0
PATCH_RADII_PX = (4, 8, 16)
UNLABELED_PER_POSITIVE = 3
MAX_TRAIN_POSITIVES_PER_VIDEO = 350
BUDGETS = (100, 250, 500, 1000)
RECALL_TOLERANCE_PX = 5.0
REGULARIZATION = 1.0
SEED = 42
SHOW_PLOTS = True


def prepare(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim == 3 and image.shape[2] == 4:
        mask = (image[:, :, 3] == 255).astype(np.uint8) * 255
        bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    else:
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        bgr = image if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    mask = cv.erode(mask, np.ones((5, 5), dtype=np.uint8))
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    return bgr, gray, mask


def propose(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    detectors = (spot_log.detect, spot_blobs.detect, spot_segmentation.detect)
    sets = [detector(gray, mask) for detector in detectors]
    points = np.concatenate(sets).astype(np.float32)

    if not len(points):
        return points.reshape(-1, 2)

    # Merge subpixel duplicates; retain representative coordinate.
    cells = np.rint(points).astype(np.int32)
    _, indices = np.unique(cells, axis=0, return_index=True)

    return points[np.sort(indices)]


def features(gray: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Small multiscale intensity/texture descriptors; no image-label leakage."""
    if not len(points):
        return np.empty((0, 6 * len(PATCH_RADII_PX)), dtype=np.float64)

    padded = cv.copyMakeBorder(
        gray, max(PATCH_RADII_PX), max(PATCH_RADII_PX),
        max(PATCH_RADII_PX), max(PATCH_RADII_PX), cv.BORDER_REFLECT_101,
    )
    enhanced = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(padded)
    rmax = max(PATCH_RADII_PX)
    vectors = []

    for x, y in points:
        x0, y0 = round(float(x)) + rmax, round(float(y)) + rmax
        descriptor = []

        for radius in PATCH_RADII_PX:
            patch = enhanced[
                y0 - radius:y0 + radius + 1,
                x0 - radius:x0 + radius + 1,
            ].astype(np.float64) / 255.0
            center = patch[radius, radius]
            mean = float(patch.mean())
            std = float(patch.std())
            edge = cv.Laplacian(patch, cv.CV_64F)
            descriptor.extend((
                center, mean, std, mean - center,
                float(np.mean(np.abs(edge))), float(np.percentile(patch, 80)),
            ))

        vectors.append(descriptor)

    return np.asarray(vectors, dtype=np.float64)


def fit_positive_unlabeled(
    positives: np.ndarray, unlabeled: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contrastive PU baseline: unlabeled are noisy references, not known negatives."""
    if not len(positives) or not len(unlabeled):
        raise ValueError("Need both positive and unlabeled training candidates")

    target_unlabeled = min(len(unlabeled), len(positives) * UNLABELED_PER_POSITIVE)
    chosen = rng.choice(len(unlabeled), target_unlabeled, replace=False)
    x = np.concatenate((positives, unlabeled[chosen]))
    y = np.concatenate((np.ones(len(positives)), np.zeros(target_unlabeled)))
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-5)
    normalized = (x - mean) / scale
    design = np.column_stack((normalized, np.ones(len(x))))
    positive_weight = 0.5 / len(positives)
    unlabeled_weight = 0.5 / target_unlabeled
    weights = np.where(y == 1, positive_weight, unlabeled_weight)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ params
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        loss = np.logaddexp(0, logits) - y * logits
        penalty = REGULARIZATION * np.sum(params[:-1] ** 2) / 2
        gradient = design.T @ (weights * (probabilities - y))
        gradient[:-1] += REGULARIZATION * params[:-1]

        return float(weights @ loss + penalty), gradient

    solution = minimize(
        objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B"
    )

    if not solution.success:
        raise RuntimeError(f"Ranking optimizer did not converge: {solution.message}")

    return solution.x, mean, scale


def load_examples(project: Project) -> list[dict]:
    examples = []

    for video_id, track_id, frame in project.metadata.list_spot_annotations():
        truth = np.asarray(
            project.metadata.read_spots(video_id, track_id, frame), dtype=np.float32
        ).reshape(-1, 2)

        if not len(truth):
            continue

        bgr, gray, mask = prepare(project.media.read_masked_crop(video_id, track_id, frame))
        proposals = propose(gray, mask)

        if not len(proposals):
            print(f"{video_id}: no detector candidates; skipping")
            continue

        distances, _ = cKDTree(truth).query(proposals)
        positive = distances <= POSITIVE_DISTANCE_PX
        unlabeled = distances >= UNLABELED_DISTANCE_PX
        descriptors = features(gray, proposals)
        examples.append({
            "video": video_id, "track": track_id, "frame": frame, "bgr": bgr,
            "truth": truth, "points": proposals, "features": descriptors,
            "positive": positive, "unlabeled": unlabeled,
        })

    return examples


def run(project: Project) -> None:
    rng = np.random.default_rng(SEED)
    examples = load_examples(project)
    videos = sorted({item["video"] for item in examples})

    if len(videos) < 2:
        raise ValueError("Need at least two annotated videos for leave-one-video-out")

    results: dict[int, list[tuple[int, int]]] = {budget: [] for budget in BUDGETS}

    for item in examples:
        training = [other for other in examples if other["video"] != item["video"]]
        positives = []
        unknowns = []

        for other in training:
            positive = other["features"][other["positive"]]
            if len(positive) > MAX_TRAIN_POSITIVES_PER_VIDEO:
                selected = rng.choice(
                    len(positive), MAX_TRAIN_POSITIVES_PER_VIDEO, replace=False
                )
                positive = positive[selected]

            positives.append(positive)
            unknowns.append(other["features"][other["unlabeled"]])

        weights, mean, scale = fit_positive_unlabeled(
            np.concatenate(positives), np.concatenate(unknowns), rng
        )
        scores = ((item["features"] - mean) / scale) @ weights[:-1] + weights[-1]
        ranked = item["points"][np.argsort(scores)[::-1]]
        baseline = item["points"]
        print(f"\nHeld out {item['video']}/{item['track']}/{item['frame']}")
        print(f"  selected annotations={len(item['truth'])}, proposals={len(ranked)}")

        for budget in BUDGETS:
            prediction = ranked[:budget]
            result = evaluate(item["truth"], prediction, RECALL_TOLERANCE_PX)
            baseline_result = evaluate(
                item["truth"], baseline[:budget], RECALL_TOLERANCE_PX
            )
            results[budget].append((len(result.matched), len(item["truth"])))
            print(
                f"  top-{budget:<4} recall={result.recall:.3f} "
                f"raw_order_baseline={baseline_result.recall:.3f}"
            )

        if SHOW_PLOTS:
            figure, axis = plt.subplots(figsize=(12, 7))
            axis.imshow(cv.cvtColor(item["bgr"], cv.COLOR_BGR2RGB))
            axis.scatter(ranked[:BUDGETS[0], 0], ranked[:BUDGETS[0], 1], s=18, c="lime")
            axis.scatter(
                item["truth"][:, 0], item["truth"][:, 1],
                s=24, facecolors="none", edgecolors="cyan",
            )
            axis.set_title(f"LOVO {item['video']}: top-{BUDGETS[0]} green, labels cyan")
            axis.axis("off")
            plt.show()
            plt.close(figure)

    print("\nMicro recall across held-out videos:")

    for budget, values in results.items():
        matched = sum(value[0] for value in values)
        labeled = sum(value[1] for value in values)
        print(f"  top-{budget:<4} {matched}/{labeled} = {matched/labeled:.3f}")

    print("Note: this is a contrastive PU heuristic, not a calibrated spot probability.")
    print("Unannotated points are not verified negatives; precision/F1 are undefined.")


if __name__ == "__main__":
    run(Project.open())
