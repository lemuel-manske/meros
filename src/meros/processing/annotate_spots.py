from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


SUBJECT_ID = "luana"
IMAGE_ID = 1


def select_spots(image: np.ndarray, initial: list[list[int]], title: str) -> list[list[int]] | None:
    points = [point.copy() for point in initial]

    figure, axis = plt.subplots(figsize=(13, 8))

    axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    axis.set_title(title)
    axis.set_axis_off()

    markers = axis.scatter([], [], s=38, facecolors="none", edgecolors="yellow")

    def refresh() -> None:
        markers.set_offsets(points if points else [[-10, -10]])

        figure.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return

        x, y = round(event.xdata), round(event.ydata)

        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            return

        if event.button == 1:
            points.append([x, y])
        elif event.button == 3 and points:
            nearest = min(
                range(len(points)),
                key=lambda index: (points[index][0] - x) ** 2 + (points[index][1] - y) ** 2,
            )

            points.pop(nearest)

        refresh()

    result: list[list[int]] | None = None

    def on_key(event) -> None:
        nonlocal result

        if event.key in ("enter", "s"):
            result = [point.copy() for point in points]

            plt.close(figure)
        elif event.key in ("escape", "q"):
            plt.close(figure)
        elif event.key in ("backspace", "ctrl+z") and points:
            points.pop()
            refresh()

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)

    refresh()
    plt.show(block=True)

    return result


def run(project: Project) -> None:
    for subject in project.media.list_subjects():
        if subject.subject_id != SUBJECT_ID:
            continue

        for photo in subject.images:
            if photo.image_id != IMAGE_ID:
                continue

            image = project.media.read_subject_image(subject.subject_id, photo.image_id)

            existing = project.metadata.read_subject_spots(subject.subject_id, photo.image_id)

            title = (
                f"{subject.subject_id} / image {photo.image_id} "
                f"({len(existing)} saved)\n"
                "Left click: add | Right click: remove nearest | "
                "Backspace: undo | Enter/S: save | Esc/Q: skip"
            )

            selected = select_spots(image, existing, title)

            if selected is None:
                continue

            project.metadata.write_subject_spots(subject.subject_id, photo.image_id, selected)

            print(f"Saved {len(selected)} spot centers for {subject.subject_id}/{photo.image_id}")


if __name__ == "__main__":
    run(Project.open())
