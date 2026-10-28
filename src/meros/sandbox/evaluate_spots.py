"""Evaluate sandbox spot detectors against manually annotated spot centers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from meros.project import Project
from meros.sandbox import spot_blobs, spot_log, spot_segmentation


# Pixel distances in masked-crop coordinates. Edit and rerun manually.
TOLERANCES_PX = (5.0, 10.0, 15.0)
DISPLAY_TOLERANCE_PX = 10.0


@dataclass(frozen=True)
class Evaluation:
    matched: list[tuple[int, int]]
    missing: list[int]
    extra: list[int]
    errors: list[float]

    @property
    def precision(self) -> float:
        total = len(self.matched) + len(self.extra)
        return len(self.matched) / total if total else 0.0

    @property
    def recall(self) -> float:
        total = len(self.matched) + len(self.missing)
        return len(self.matched) / total if total else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


def evaluate(truth: np.ndarray, predicted: np.ndarray, tolerance: float) -> Evaluation:
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    if len(truth) == 0 or len(predicted) == 0:
        return Evaluation([], list(range(len(truth))), list(range(len(predicted))), [])

    distances = cdist(truth, predicted)
    # Each reference point may match one prediction or its own dummy column.
    # Dummy cost ensures maximum valid matches before minimizing distance.
    penalty = tolerance + 1.0
    costs = np.full((len(truth), len(predicted) + len(truth)), penalty)
    costs[:, : len(predicted)] = np.where(distances <= tolerance, distances, 2 * penalty)

    rows, cols = linear_sum_assignment(costs)
    matched = [
        (int(i), int(j))
        for i, j in zip(rows, cols, strict=True)
        if j < len(predicted) and distances[i, j] <= tolerance
    ]
    matched_truth = {i for i, _ in matched}
    matched_predicted = {j for _, j in matched}

    return Evaluation(
        matched,
        [i for i in range(len(truth)) if i not in matched_truth],
        [j for j in range(len(predicted)) if j not in matched_predicted],
        [float(distances[i, j]) for i, j in matched],
    )


def plot_results(
    image: np.ndarray,
    truth: np.ndarray,
    detections: dict[str, np.ndarray],
    evaluations: dict[str, Evaluation],
    title: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    original = axes[0, 0]
    original.imshow(rgb)
    original.set_title(f"Manual ground truth: {len(truth)} points")

    if len(truth):
        original.scatter(truth[:, 0], truth[:, 1], s=18, c="cyan")

    for axis, (name, predicted) in zip(list(axes.flat)[1:], detections.items(), strict=True):
        result = evaluations[name]
        axis.imshow(rgb)
        good = [predicted[j] for _, j in result.matched]
        extra = [predicted[j] for j in result.extra]
        missing = [truth[i] for i in result.missing]

        for points, color, marker in (
            (good, "lime", "o"),
            (extra, "red", "x"),
            (missing, "cyan", "+"),
        ):
            if points:
                coords = np.asarray(points)
                axis.scatter(coords[:, 0], coords[:, 1], c=color, marker=marker, s=25)

        axis.set_title(
            f"{name}: P={result.precision:.3f}, R={result.recall:.3f}, F1={result.f1:.3f}"
        )

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(title + " | green=TP, red=FP, cyan=FN")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def run(project: Project) -> None:
    detectors = {
        "LoG": spot_log.detect,
        "Blobs": spot_blobs.detect,
        "Segmentation": spot_segmentation.detect,
    }
    totals: dict[float, dict[str, dict[str, object]]] = {
        tolerance: {
            name: {"tp": 0, "fp": 0, "fn": 0, "errors": []} for name in detectors
        }
        for tolerance in TOLERANCES_PX
    }
    count = 0

    for video_id, track_id, frame in project.metadata.list_spot_annotations():
        image = project.media.read_masked_crop(video_id, track_id, frame)

        if image.ndim == 3 and image.shape[2] == 4:
            mask = (image[:, :, 3] == 255).astype(np.uint8) * 255
            bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        else:
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            bgr = image if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        mask = cv.erode(mask, np.ones((5, 5), dtype=np.uint8))
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        truth = np.asarray(
            project.metadata.read_spots(video_id, track_id, frame), dtype=np.float32
        ).reshape(-1, 2)
        detected = {name: detector(gray, mask) for name, detector in detectors.items()}
        preview: dict[str, Evaluation] = {}

        print(f"\n{video_id}/{track_id}/{frame}: {len(truth)} annotated spots")

        for tolerance in TOLERANCES_PX:
            print(f"  Tolerance: {tolerance:g}px")

            for name, predicted in detected.items():
                result = evaluate(truth, predicted, tolerance)
                stats = totals[tolerance][name]
                stats["tp"] = int(stats["tp"]) + len(result.matched)
                stats["fp"] = int(stats["fp"]) + len(result.extra)
                stats["fn"] = int(stats["fn"]) + len(result.missing)
                errors = stats["errors"]
                assert isinstance(errors, list)
                errors.extend(result.errors)
                print(
                    f"    {name:<14} detected={len(predicted):4} "
                    f"TP={len(result.matched):4} FP={len(result.extra):4} "
                    f"FN={len(result.missing):4} F1={result.f1:.3f}"
                )

                if tolerance == DISPLAY_TOLERANCE_PX:
                    preview[name] = result

        if len(preview) == len(detectors):
            plot_results(bgr, truth, detected, preview, f"{video_id}/{track_id}/{frame}")

        count += 1

    if not count:
        print("No manual spot annotations found.")
        return

    print(f"\nAggregate results over {count} annotated images (micro averages):")

    for tolerance, methods in totals.items():
        print(f"  Tolerance: {tolerance:g}px")

        for name, stats in methods.items():
            tp = int(stats["tp"])
            fp = int(stats["fp"])
            fn = int(stats["fn"])
            errors = stats["errors"]
            assert isinstance(errors, list)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            mae = float(np.mean(errors)) if errors else float("nan")
            print(
                f"    {name:<14} P={precision:.3f} R={recall:.3f} "
                f"F1={f1:.3f} mean_error={mae:.2f}px"
            )


if __name__ == "__main__":
    run(Project.open())
