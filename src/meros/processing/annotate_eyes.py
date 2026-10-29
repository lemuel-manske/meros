from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

from meros.domain import EyeAnnotation
from meros.project import Project


def annotate_eye(
    image: np.ndarray, existing: EyeAnnotation | None, title: str
) -> EyeAnnotation | None:
    figure, axis = plt.subplots(figsize=(13, 8))

    axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    axis.set_title(title)
    axis.set_axis_off()

    rectangle = Rectangle((0, 0), 0, 0, edgecolor="cyan", facecolor="none", linewidth=2)
    ellipse = Ellipse(
        (0, 0), 0, 0, edgecolor="yellow", facecolor="none", linewidth=2
    )

    axis.add_patch(rectangle)
    axis.add_patch(ellipse)

    result: EyeAnnotation | None = None

    selection = existing

    start: tuple[float, float] | None = None

    dragging = False

    def paint() -> None:
        if selection is None:
            ellipse.set_visible(False)
        else:
            ellipse.set_visible(True)

            ellipse.center = selection.center
            ellipse.width = 2 * selection.radius_x
            ellipse.height = 2 * selection.radius_y
            ellipse.angle = selection.angle_degrees

        figure.canvas.draw_idle()

    def on_press(event) -> None:
        nonlocal start, dragging

        if event.inaxes is axis and event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                start = (event.xdata, event.ydata)

                dragging = True

                rectangle.set_visible(True)
                rectangle.set_xy(start)
                rectangle.set_width(0)
                rectangle.set_height(0)

    def on_motion(event) -> None:
        if not dragging or start is None or event.inaxes is not axis:
            return

        if event.xdata is None or event.ydata is None:
            return

        rectangle.set_xy((min(start[0], event.xdata), min(start[1], event.ydata)))
        rectangle.set_width(abs(event.xdata - start[0]))
        rectangle.set_height(abs(event.ydata - start[1]))

        figure.canvas.draw_idle()

    def on_release(event) -> None:
        nonlocal selection, dragging, start

        if not dragging or start is None:
            return

        dragging = False
        rectangle.set_visible(False)

        if event.inaxes is axis and event.xdata is not None and event.ydata is not None:
            x1 = float(np.clip(start[0], 0, image.shape[1] - 1))
            y1 = float(np.clip(start[1], 0, image.shape[0] - 1))
            x2 = float(np.clip(event.xdata, 0, image.shape[1] - 1))
            y2 = float(np.clip(event.ydata, 0, image.shape[0] - 1))

            rx, ry = abs(x2 - x1) / 2, abs(y2 - y1) / 2

            if rx >= 2 and ry >= 2:
                selection = EyeAnnotation(
                    center=((x1 + x2) / 2, (y1 + y2) / 2),
                    radius_x=rx,
                    radius_y=ry,
                )

        start = None
        paint()

    def on_key(event) -> None:
        nonlocal result, selection

        if event.key in ("enter", "s"):
            if selection is not None:
                result = selection
                plt.close(figure)
        elif event.key in ("escape", "q"):
            plt.close(figure)
        elif event.key in ("backspace", "delete", "r"):
            selection = None
            paint()

    figure.canvas.mpl_connect("button_press_event", on_press)
    figure.canvas.mpl_connect("motion_notify_event", on_motion)
    figure.canvas.mpl_connect("button_release_event", on_release)
    figure.canvas.mpl_connect("key_press_event", on_key)

    rectangle.set_visible(False)

    paint()
    plt.show(block=True)

    return result


def run(project: Project) -> None:
    count = 0

    for subject in project.media.list_subjects():
        for photo in subject.images:
            image = project.media.read_subject_image(subject.subject_id, photo.image_id)

            existing = project.metadata.read_subject_eye(subject.subject_id, photo.image_id)

            title = (
                f"{subject.subject_id} / {photo.image_id} (left eye)\n"
                "Drag: outline eye | Enter/S: save | R/Backspace: redraw | Esc/Q: skip"
            )

            selected = annotate_eye(image, existing, title)

            if selected is None:
                print(f"Skipped {subject.subject_id}/{photo.image_id}")
                continue

            project.metadata.write_subject_eye(subject.subject_id, photo.image_id, selected)

            count += 1

            print(f"Saved left eye ellipse for {subject.subject_id}/{photo.image_id}")

    print(f"Saved {count} eye annotations.")


if __name__ == "__main__":
    run(Project.open())
