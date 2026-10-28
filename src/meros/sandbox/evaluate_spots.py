"""Evaluate spot detectors near manually selected spot centers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from meros.project import Project
from meros.sandbox import spot_blobs, spot_log, spot_segmentation


TOLERANCES_PX = (5.0, 10.0, 15.0)

DISPLAY_TOLERANCE_PX = 10.0

ROI_RADIUS_PX = 25  # Neighborhood around each manually selected spot.
ROI_CLOSING_PX = 9  # Join nearby neighborhoods without filling the whole head.


@dataclass(frozen=True)
class Evaluation:
    matched: list[tuple[int, int]]
    missing: list[int]
    extra: list[int]
    errors: list[float]

    @property
    def recall(self) -> float:
        total = len(self.matched) + len(self.missing)
        return len(self.matched) / total if total else 0.0


def evaluation_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Use only areas supported by annotated spots, not the entire fish."""
    roi = np.zeros(shape, dtype=np.uint8)

    for x, y in points:
        cv.circle(roi, (round(float(x)), round(float(y))), ROI_RADIUS_PX, 255, -1)

    if ROI_CLOSING_PX > 1:
        radius = ROI_CLOSING_PX
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (radius, radius))
        roi = cv.morphologyEx(roi, cv.MORPH_CLOSE, kernel)

    return roi


def inside(points: np.ndarray, roi: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points

    x = np.rint(points[:, 0]).astype(int)
    y = np.rint(points[:, 1]).astype(int)
    valid = (x >= 0) & (y >= 0) & (x < roi.shape[1]) & (y < roi.shape[0])
    selected = np.zeros(len(points), dtype=bool)
    selected[valid] = roi[y[valid], x[valid]] != 0

    return points[selected]


def evaluate(truth: np.ndarray, predicted: np.ndarray, tolerance: float) -> Evaluation:
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    if not len(truth) or not len(predicted):
        return Evaluation([], list(range(len(truth))), list(range(len(predicted))), [])

    distances = cdist(truth, predicted)
    penalty = tolerance + 1.0
    costs = np.full((len(truth), len(predicted) + len(truth)), penalty)
    costs[:, :len(predicted)] = np.where(distances <= tolerance, distances, 2 * penalty)
    rows, cols = linear_sum_assignment(costs)
    matched = [
        (int(i), int(j))
        for i, j in zip(rows, cols, strict=True)
        if j < len(predicted) and distances[i, j] <= tolerance
    ]
    selected_truth = {i for i, _ in matched}
    selected_predicted = {j for _, j in matched}

    return Evaluation(
        matched,
        [i for i in range(len(truth)) if i not in selected_truth],
        [j for j in range(len(predicted)) if j not in selected_predicted],
        [float(distances[i, j]) for i, j in matched],
    )


def display(
    bgr: np.ndarray,
    truth: np.ndarray,
    roi: np.ndarray,
    predictions: dict[str, np.ndarray],
    comparisons: dict[str, Evaluation],
    title: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    for axis in axes.flat:
        axis.imshow(rgb)
        axis.contour(roi, levels=[127], colors=["yellow"], linewidths=1)
        axis.axis("off")

    original = axes[0, 0]
    original.set_title(f"Selected annotations: {len(truth)} | ROI outlined in yellow")

    if len(truth):
        original.scatter(truth[:, 0], truth[:, 1], c="cyan", s=16)

    for axis, (name, predicted) in zip(list(axes.flat)[1:], predictions.items(), strict=True):
        result = comparisons[name]
        matched = [predicted[j] for _, j in result.matched]
        unmatched = [predicted[j] for j in result.extra]
        missed = [truth[i] for i in result.missing]

        for points, color, marker in (
            (matched, "lime", "o"),
            (unmatched, "red", "x"),
            (missed, "cyan", "+"),
        ):
            if points:
                coords = np.asarray(points)
                axis.scatter(coords[:, 0], coords[:, 1], c=color, marker=marker, s=22)

        axis.set_title(
            f"{name}: selected recall={result.recall:.3f}, candidates={len(predicted)}"
        )

    figure.suptitle(title + " | green=matched, red=unlabeled, cyan=missed")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def run(project: Project) -> None:
    detectors = {
        "LoG": spot_log.detect,
        "Blobs": spot_blobs.detect,
        "Segmentation": spot_segmentation.detect,
    }
    totals = {
        tolerance: {
            name: {"matched": 0, "missed": 0, "candidates": 0, "errors": []}
            for name in detectors
        }
        for tolerance in TOLERANCES_PX
    }
    count = 0

    for video_id, track_id, frame in project.metadata.list_spot_annotations():
        truth = np.asarray(
            project.metadata.read_spots(video_id, track_id, frame), dtype=np.float32
        ).reshape(-1, 2)

        if not len(truth):
            print(f"Skipping {video_id}/{track_id}/{frame}: no selected spots")
            continue

        image = project.media.read_masked_crop(video_id, track_id, frame)

        if image.ndim == 3 and image.shape[2] == 4:
            mask = (image[:, :, 3] == 255).astype(np.uint8) * 255
            bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        else:
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            bgr = image if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        mask = cv.erode(mask, np.ones((5, 5), dtype=np.uint8))
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        roi = cv.bitwise_and(evaluation_mask(gray.shape, truth), mask)
        selected_truth = inside(truth, roi)
        detections = {
            name: inside(detector(gray, mask), roi)
            for name, detector in detectors.items()
        }

        print(f"\n{video_id}/{track_id}/{frame}: {len(selected_truth)} selected spots")
        preview: dict[str, Evaluation] = {}

        for tolerance in TOLERANCES_PX:
            print(f"  Tolerance: {tolerance:g}px")

            for name, predicted in detections.items():
                result = evaluate(selected_truth, predicted, tolerance)
                stats = totals[tolerance][name]
                stats["matched"] += len(result.matched)
                stats["missed"] += len(result.missing)
                stats["candidates"] += len(predicted)
                stats["errors"].extend(result.errors)
                print(
                    f"    {name:<14} candidates={len(predicted):4} "
                    f"matched={len(result.matched):4} missed={len(result.missing):4} "
                    f"selected_recall={result.recall:.3f}"
                )

                if tolerance == DISPLAY_TOLERANCE_PX:
                    preview[name] = result

        if len(preview) == len(detectors):
            display(
                bgr, selected_truth, roi, detections, preview,
                f"{video_id}/{track_id}/{frame}",
            )

        count += 1

    if not count:
        print("No nonempty spot annotations found.")
        return

    print(f"\nAggregate results over {count} annotated images:")

    for tolerance, methods in totals.items():
        print(f"  Tolerance: {tolerance:g}px")

        for name, stats in methods.items():
            matched = stats["matched"]
            missed = stats["missed"]
            candidates = stats["candidates"]
            errors = stats["errors"]
            recall = matched / (matched + missed) if matched + missed else 0.0
            mean_error = float(np.mean(errors)) if errors else float("nan")
            print(
                f"    {name:<14} selected_recall={recall:.3f} "
                f"candidates={candidates} mean_error={mean_error:.2f}px"
            )


if __name__ == "__main__":
    run(Project.open())
