"""Compare spot annotation scales and detector coverage without training a model."""

from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree

from meros.project import Project
from meros.sandbox import spot_blobs, spot_log, spot_segmentation
from meros.sandbox.evaluate_spots import evaluate
from meros.sandbox.spot_pu_ranker import prepare


# Directly editable sandbox settings.
MATCH_TOLERANCE_PX = 5.0
CANDIDATE_BUDGETS_PER_100_LABELS = (25, 50, 100, 200)
MIN_CANDIDATE_BUDGET = 25
SHOW_PLOTS = True


def spacing(points: np.ndarray) -> np.ndarray:
    """Nearest-neighbor center distances in pixels; density proxy, not physical spot size."""
    if len(points) < 2:
        return np.empty(0, dtype=np.float64)

    distances, _ = cKDTree(points).query(points, k=2)

    return np.asarray(distances[:, 1], dtype=np.float64)


def summarize(name: str, values: np.ndarray) -> str:
    if not len(values):
        return f"{name}: N/A"

    p10, p50, p90 = np.percentile(values, (10, 50, 90))

    return f"{name}: p10={p10:.1f}, median={p50:.1f}, p90={p90:.1f}"


def propose(gray: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "LoG": spot_log.detect(gray, mask),
        "Blobs": spot_blobs.detect(gray, mask),
        "Segmentation": spot_segmentation.detect(gray, mask),
    }


def run(project: Project) -> None:
    per_method: dict[str, dict[int, list[float]]] = {
        name: {rate: [] for rate in CANDIDATE_BUDGETS_PER_100_LABELS}
        for name in ("LoG", "Blobs", "Segmentation")
    }
    included = 0

    for video_id, track_id, frame in project.metadata.list_spot_annotations():
        truth = np.asarray(
            project.metadata.read_spots(video_id, track_id, frame), dtype=np.float32
        ).reshape(-1, 2)

        if not len(truth):
            print(f"{video_id}/{track_id}/{frame}: no annotations; excluded")
            continue

        image = project.media.read_masked_crop(video_id, track_id, frame)
        bgr, gray, mask = prepare(image)
        distances = spacing(truth)
        area = int(np.count_nonzero(mask))
        density = len(truth) / (area / 10000.0) if area else float("nan")
        levels = np.asarray(gray[mask != 0], dtype=np.float64)
        contrast = float(np.std(levels)) if len(levels) else float("nan")
        candidates = propose(gray, mask)

        print(f"\n{video_id}/{track_id}/{frame}")
        print(f"  selected labels={len(truth)}, fish pixels={area}")
        print(f"  selected labels / 10k fish pixels={density:.2f} (NOT full spot density)")
        print(f"  grayscale standard deviation={contrast:.2f}")
        print(f"  {summarize('nearest annotated center distance (px)', distances)}")

        for name, predicted in candidates.items():
            for rate in CANDIDATE_BUDGETS_PER_100_LABELS:
                budget = max(MIN_CANDIDATE_BUDGET, round(rate * len(truth) / 100))
                # The original detector's order is not a confidence ranking.
                result = evaluate(truth, predicted[:budget], MATCH_TOLERANCE_PX)
                per_method[name][rate].append(result.recall)

                print(
                    f"  {name:<14} budget={budget:<5} ({rate}/100 annotations) "
                    f"selected_recall={result.recall:.3f}"
                )

        if SHOW_PLOTS:
            figure, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
            axes[0].scatter(truth[:, 0], truth[:, 1], s=8, c="cyan")
            axes[0].set_title(f"{video_id}: {len(truth)} selected centers")
            axes[1].hist(distances, bins=20)
            axes[1].set_xlabel("Nearest annotated neighbor (px)")
            axes[1].set_ylabel("Count")

            for axis in axes:
                if axis is axes[0]:
                    axis.axis("off")

            figure.tight_layout()
            plt.show()
            plt.close(figure)

        included += 1

    if not included:
        print("No annotated frames with positive spots available.")
        return

    print(f"\nMacro recall over {included} annotated images (each image equal weight):")

    for name, by_rate in per_method.items():
        for rate, recalls in by_rate.items():
            print(
                f"  {name:<14} budget={rate:3}/100 labels "
                f"macro_recall={np.mean(recalls):.3f}"
            )

    print("Important: annotations are incomplete, and detector order is not calibrated.")
    print("Pixel spacing and annotation density are not physical spot diameter or true density.")


if __name__ == "__main__":
    run(Project.open())
