"""One-factor-at-a-time sweep for adaptive LoG on partial spot labels.

Compares candidate coverage separately from top-K ordering. No training and no
automatic parameter selection: review qualitative noise before choosing settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from meros.project import Project
from meros.sandbox.evaluate_spots import evaluate
from meros.sandbox.spot_log import detect as original_log
from meros.sandbox import spot_log_adaptive as adaptive


# Edit these exploratory settings directly.
BUDGETS = (100, 250, 500, 1000)
TOLERANCE_PX = 5.0
RESPONSE_VALUES = (0.6, 0.9, 1.2)
CONTRAST_VALUES = (0.0, 0.05, 0.10)
NMS_VALUES = (0.4, 0.6, 0.8)
MAX_CANDIDATES_FOR_REPORT = 10000


@dataclass
class Crop:
    name: str
    truth: np.ndarray
    gray: np.ndarray
    mask: np.ndarray


def load(project: Project) -> list[Crop]:
    crops = []

    for video_id, track_id, frame in project.metadata.list_spot_annotations():
        truth = np.asarray(
            project.metadata.read_spots(video_id, track_id, frame), dtype=np.float32
        ).reshape(-1, 2)

        if not len(truth):
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
        crops.append(Crop(f"{video_id}/{track_id}/{frame}", truth, gray, mask))

    return crops


def configs() -> list[tuple[str, dict[str, float]]]:
    base = {
        "min_response": adaptive.MIN_RESPONSE,
        "min_relative_contrast": adaptive.MIN_RELATIVE_CONTRAST,
        "nms_radius_factor": adaptive.NMS_RADIUS_FACTOR,
    }
    options = [("current", base)]

    for key, values in (
        ("min_response", RESPONSE_VALUES),
        ("min_relative_contrast", CONTRAST_VALUES),
        ("nms_radius_factor", NMS_VALUES),
    ):
        for value in values:
            if value == base[key]:
                continue

            changed = {**base, key: value}
            options.append((f"{key}={value:g}", changed))

    return options


def summarize(crops: list[Crop], name: str, params: dict[str, float] | None) -> None:
    recalls = {budget: [] for budget in BUDGETS}
    coverage = []
    counts = []
    print(f"\n{name}")

    for crop in crops:
        if params is None:
            predicted = original_log(crop.gray, crop.mask)
        else:
            predicted, _, _ = adaptive.detect_with_details(
                crop.gray, crop.mask, **params
            )

        counts.append(len(predicted))
        full = evaluate(crop.truth, predicted, TOLERANCE_PX).recall
        coverage.append(full)
        for budget in BUDGETS:
            recalls[budget].append(
                evaluate(crop.truth, predicted[:budget], TOLERANCE_PX).recall
            )

        print(
            f"  {crop.name:<17} candidates={len(predicted):5} "
            f"full_recall={full:.3f} "
            + " ".join(
                f"top{budget}={recalls[budget][-1]:.3f}" for budget in BUDGETS
            )
        )

    summary = " ".join(
        f"top{budget}={np.mean(recalls[budget]):.3f}" for budget in BUDGETS
    )
    print(
        f"  MACRO candidates={np.mean(counts):.1f} "
        f"full_recall={np.mean(coverage):.3f} {summary}"
    )


def run(project: Project) -> None:
    crops = load(project)
    if not crops:
        print("No annotated spot centers available.")
        return

    print(f"Comparing {len(crops)} images, tolerance={TOLERANCE_PX:g}px")
    print(
        "Changing only one adaptive parameter at a time. "
        "The default is shown first."
    )
    summarize(crops, "Original LoG", None)

    for name, params in configs():
        summarize(crops, "Adaptive " + name, params)

    print(
        "\nInterpretation: full_recall measures coverage of selected labels. "
        "No precision is available because labels are partial."
    )
    print(
        "High recall with many candidates does not imply fewer eye/mouth errors. "
        "Use compare_log.py and visual review before accepting a configuration."
    )


if __name__ == "__main__":
    run(Project.open())
