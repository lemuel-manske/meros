"""Review proposed spot matches one at a time without changing the matcher."""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.processing.compare_joint_spot_matching import match
from meros.processing.compare_spot_geometry import load
from meros.project import Project


SUBJECT_ID = "luana"
FIRST_IMAGE_ID = 0
SECOND_IMAGE_ID = 1


def review(
    first_image: np.ndarray,
    second_image: np.ndarray,
    pairs: list[tuple[list[int], list[int]]],
    existing: list[dict],
) -> list[dict] | None:
    previous = {
        (tuple(item["first"]), tuple(item["second"])): item["label"]
        for item in existing
    }
    labels = [previous.get((tuple(a), tuple(b))) for a, b in pairs]

    figure, axes = plt.subplots(1, 2, figsize=(17, 8))

    for axis, image, image_id in zip(
        axes, (first_image, second_image), (FIRST_IMAGE_ID, SECOND_IMAGE_ID),
        strict=True,
    ):
        axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axis.set_title(f"{SUBJECT_ID}/{image_id}")
        axis.set_axis_off()

    markers = [
        axis.scatter([], [], s=110, facecolors="none", edgecolors="yellow", linewidths=2)
        for axis in axes
    ]

    index = 0
    saved = False

    def paint() -> None:
        for marker, point in zip(markers, pairs[index], strict=True):
            marker.set_offsets([point])

        counts = {
            label: labels.count(label)
            for label in ("correct", "incorrect", "uncertain")
        }
        figure.suptitle(
            f"Pair {index + 1}/{len(pairs)} — "
            f"Current: {labels[index] or 'unreviewed'}\n"
            "1: correct | 2: incorrect | 3: uncertain | "
            "Left/Right: navigate | Enter/S: save | Esc/Q: cancel\n"
            f"Correct {counts['correct']}  Incorrect {counts['incorrect']}  "
            f"Uncertain {counts['uncertain']}"
        )

        figure.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal index, saved

        if event.key in ("1", "2", "3"):
            labels[index] = {
                "1": "correct", "2": "incorrect", "3": "uncertain"
            }[event.key]

            if index < len(pairs) - 1:
                index += 1

        elif event.key == "right":
            index = min(index + 1, len(pairs) - 1)

        elif event.key == "left":
            index = max(index - 1, 0)

        elif event.key in ("enter", "s"):
            saved = True
            plt.close(figure)
            return

        elif event.key in ("escape", "q"):
            plt.close(figure)
            return

        paint()

    figure.canvas.mpl_connect("key_press_event", on_key)
    paint()
    plt.show(block=True)

    if not saved:
        return None

    return [
        {"first": a, "second": b, "label": label}
        for (a, b), label in zip(pairs, labels, strict=True)
        if label is not None
    ]


def run(project: Project) -> None:
    samples = load(project)
    first = next(
        sample for sample in samples
        if (sample.subject_id, sample.image_id) == (SUBJECT_ID, FIRST_IMAGE_ID)
    )
    second = next(
        sample for sample in samples
        if (sample.subject_id, sample.image_id) == (SUBJECT_ID, SECOND_IMAGE_ID)
    )

    indices = match(first.normalized, second.normalized)
    pairs = [
        (
            [int(round(value)) for value in first.spots[i]],
            [int(round(value)) for value in second.spots[j]],
        )
        for i, j in indices
    ]

    if not pairs:
        print("No proposed spot correspondences.")
        return

    existing = project.metadata.read_subject_spot_reviews(
        SUBJECT_ID, FIRST_IMAGE_ID, SECOND_IMAGE_ID
    )
    reviewed = review(
        project.media.read_subject_image(SUBJECT_ID, FIRST_IMAGE_ID),
        project.media.read_subject_image(SUBJECT_ID, SECOND_IMAGE_ID),
        pairs,
        existing,
    )

    if reviewed is None:
        print("Review cancelled without saving.")
        return

    project.metadata.write_subject_spot_reviews(
        SUBJECT_ID, FIRST_IMAGE_ID, SECOND_IMAGE_ID, reviewed
    )

    print(f"Saved {len(reviewed)}/{len(pairs)} reviewed correspondences.")
    for label in ("correct", "incorrect", "uncertain"):
        print(f"  {label}: {sum(item['label'] == label for item in reviewed)}")


if __name__ == "__main__":
    run(Project.open())
