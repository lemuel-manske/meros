"""Annotate corresponding spots in two photographs of one subject."""

from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


SUBJECT_ID = None


def annotate_pair(
    subject_id: str,
    first_image_id: int,
    second_image_id: int,
    first_image: np.ndarray,
    second_image: np.ndarray,
    existing: list[dict],
    title: str,
) -> list[dict] | None:
    pairs = [{"first": list(pair["first"]), "second": list(pair["second"])} for pair in existing]
    pending: list[int] | None = None
    result: list[dict] | None = None

    figure, axes = plt.subplots(1, 2, figsize=(17, 8))
    first_axis, second_axis = axes

    for axis, image, image_id in zip(
        axes,
        (first_image, second_image),
        (first_image_id, second_image_id),
        strict=True,
    ):
        axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axis.set_title(f"{subject_id}/{image_id}")
        axis.set_axis_off()

    figure.suptitle(title)

    markers = [axis.scatter([], [], s=38, facecolors="none", edgecolors="yellow") for axis in axes]
    labels: list = []
    pending_marker = first_axis.scatter([], [], s=100, marker="x", c="cyan", linewidths=2)

    def refresh() -> None:
        nonlocal labels

        for label in labels:
            label.remove()

        labels = []

        for index, axis in enumerate(axes):
            field = "first" if index == 0 else "second"
            points = [pair[field] for pair in pairs]

            markers[index].set_offsets(points if points else [[-10, -10]])

            for number, point in enumerate(points, start=1):
                labels.append(
                    axis.annotate(
                        str(number),
                        (point[0], point[1]),
                        xytext=(6, -6),
                        textcoords="offset points",
                        color="yellow",
                        fontsize=10,
                        fontweight="bold",
                    )
                )

        pending_marker.set_offsets([pending] if pending is not None else [[-10, -10]])

        first_axis.set_title(
            f"{subject_id}/{first_image_id}"
            + (" — select matching spot on right" if pending is not None else "")
        )
        figure.canvas.draw_idle()

    def on_click(event) -> None:
        nonlocal pending

        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        if pending is None and event.inaxes is first_axis:
            pending = [round(event.xdata), round(event.ydata)]

        elif pending is not None and event.inaxes is second_axis:
            right_point = [round(event.xdata), round(event.ydata)]
            pairs.append({"first": pending, "second": right_point})
            pending = None

        refresh()

    def on_key(event) -> None:
        nonlocal pending, result

        if event.key in ("backspace", "ctrl+z", "r"):
            if pending is not None:
                pending = None

            elif pairs:
                pairs.pop()

            refresh()

        elif event.key in ("enter", "s"):
            if pending is None:
                result = [
                    {"first": pair["first"].copy(), "second": pair["second"].copy()}
                    for pair in pairs
                ]
                plt.close(figure)

        elif event.key in ("escape", "q"):
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)

    refresh()
    plt.show(block=True)

    return result


def run(project: Project) -> None:
    first_image_id = 0

    for subject in project.media.list_subjects():
        if SUBJECT_ID is not None and subject.subject_id != SUBJECT_ID:
            continue

        first_image = project.media.read_subject_image(subject.subject_id, first_image_id)

        m = subject.images[1:]

        if len(m) > 0: # not a single image subject
            print(f"Annotating subject: {subject.subject_id}, with images count:", len(m))

        for photo in m:
            second_image_id = photo.image_id

            second_image = project.media.read_subject_image(subject.subject_id, second_image_id)

            existing = project.metadata.read_subject_spot_correspondences(
                subject.subject_id, first_image_id, second_image_id
            )

            selected = annotate_pair(
                subject.subject_id,
                first_image_id,
                second_image_id,
                first_image,
                second_image,
                existing,
                (
                    "Click a spot on the left, then the SAME spot on the right. "
                    "Repeat to add numbered pairs.\n"
                    "Backspace/R: undo | Enter/S: save | Esc/Q: skip"
                ),
            )

            if selected is None:
                print("Skipped without saving.")
                return

            project.metadata.write_subject_spot_correspondences(
                subject.subject_id, first_image_id, second_image_id, selected
            )

            print(
                f"Saved {len(selected)} correspondences for {subject.subject_id}/{first_image_id}-{second_image_id}"
            )


if __name__ == "__main__":
    run(Project.open())
