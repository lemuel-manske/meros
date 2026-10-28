"""Compare original and adaptive LoG fairly at identical candidate budgets."""

from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project
from meros.sandbox.evaluate_spots import evaluate
from meros.sandbox.spot_log import detect as original_log
from meros.sandbox.spot_log_adaptive import detect_with_details


# Edit in the sandbox, no CLI arguments.
BUDGETS = (100, 250, 500, 1000)
TOLERANCE_PX = 5.0
SHOW_PLOTS = True


def run(project: Project) -> None:
    scores = {method: {budget: [] for budget in BUDGETS} for method in ("LoG", "Adaptive")}
    analyzed = 0

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
        original = original_log(gray, mask)
        adaptive, scales, strengths = detect_with_details(gray, mask)
        predictions = {"LoG": original, "Adaptive": adaptive}

        print(
            f"\n{video_id}/{track_id}/{frame}: labels={len(truth)} "
            f"original={len(original)} adaptive={len(adaptive)}"
        )

        for budget in BUDGETS:
            parts = []

            for name, candidates in predictions.items():
                result = evaluate(truth, candidates[:budget], TOLERANCE_PX)
                scores[name][budget].append(result.recall)
                parts.append(f"{name}={result.recall:.3f}")

            print(f"  top-{budget}: " + " | ".join(parts))

        if SHOW_PLOTS:
            figure, axes = plt.subplots(1, 2, figsize=(14, 6))

            for axis, (name, candidates) in zip(axes, predictions.items(), strict=True):
                axis.imshow(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
                selected = candidates[:BUDGETS[0]]

                if len(selected):
                    axis.scatter(selected[:, 0], selected[:, 1], s=12, c="yellow")

                axis.set_title(f"{name} | top-{BUDGETS[0]}")
                axis.axis("off")

            figure.suptitle(f"{video_id}/{track_id}/{frame} | labels={len(truth)}")
            figure.tight_layout()
            plt.show()
            plt.close(figure)

        if len(scales):
            print(
                f"  Adaptive sigma median={np.median(scales):.2f}, "
                f"score median={np.median(strengths):.2f}"
            )

        analyzed += 1

    if not analyzed:
        print("No annotated images with spots were available.")
        return

    print(f"\nMacro recall over {analyzed} images, {TOLERANCE_PX:g}px tolerance:")

    for budget in BUDGETS:
        original = float(np.mean(scores["LoG"][budget]))
        adaptive = float(np.mean(scores["Adaptive"][budget]))
        print(
            f"  top-{budget}: original={original:.3f} "
            f"adaptive={adaptive:.3f} delta={adaptive - original:+.3f}"
        )

    print("Unlabeled detections are not confirmed false positives.")
    print("No labels were used to select or scale candidates at inference.")


if __name__ == "__main__":
    run(Project.open())
