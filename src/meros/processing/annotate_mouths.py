"""Mark the posterior mouth commissure on left-side subject photographs."""

from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.project import Project


SUBJECT_ID: str | None = None
IMAGE_ID: int | None = None
ONLY_WITH_EYE_AND_SPOTS = True


def select_mouth(
    image: np.ndarray,
    existing: tuple[float, float] | None,
    eye_center: tuple[float, float] | None,
    spots: list[list[int]],
    title: str,
) -> tuple[float, float] | None:
    figure, axis = plt.subplots(figsize=(13, 8))
    axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    axis.set_title(title)
    axis.axis("off")

    if eye_center is not None:
        axis.scatter([eye_center[0]], [eye_center[1]], c="cyan", s=70, marker="+")

    if spots:
        markers = np.asarray(spots)
        axis.scatter(markers[:, 0], markers[:, 1], s=7, c="yellow", alpha=0.45)

    point = existing
    result: tuple[float, float] | None = None
    marker = axis.scatter([], [], s=90, c="red", marker="x")

    def refresh() -> None:
        marker.set_offsets([point] if point is not None else [[-10, -10]])
        figure.canvas.draw_idle()

    def on_click(event) -> None:
        nonlocal point

        if event.inaxes is not axis or event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        point = (float(event.xdata), float(event.ydata))
        refresh()

    def on_key(event) -> None:
        nonlocal point, result

        if event.key in ("enter", "s") and point is not None:
            result = point
            plt.close(figure)

        elif event.key in ("escape", "q"):
            plt.close(figure)

        elif event.key in ("backspace", "delete", "r"):
            point = None
            refresh()

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)

    refresh()
    plt.show(block=True)

    return result


def run(project: Project) -> None:
    saved = 0

    for subject in project.media.list_subjects():
        if SUBJECT_ID is not None and subject.subject_id != SUBJECT_ID:
            continue

        for photo in subject.images:
            if IMAGE_ID is not None and photo.image_id != IMAGE_ID:
                continue

            spots = project.metadata.read_subject_spots(subject.subject_id, photo.image_id)
            eye = project.metadata.read_subject_eye(subject.subject_id, photo.image_id)

            if ONLY_WITH_EYE_AND_SPOTS and (not spots or eye is None):
                continue

            image = project.media.read_subject_image(subject.subject_id, photo.image_id)
            existing = project.metadata.read_subject_mouth(subject.subject_id, photo.image_id)

            title = (
                f"{subject.subject_id}/{photo.image_id} — posterior mouth commissure\n"
                "Left click: mark | Enter/S: save | R/Backspace: reset | Esc/Q: skip"
            )

            selected = select_mouth(
                image,
                existing,
                eye.center if eye is not None else None,
                spots,
                title,
            )

            if selected is None:
                continue

            project.metadata.write_subject_mouth(
                subject.subject_id, photo.image_id, selected
            )
            saved += 1
            print(f"Saved mouth landmark for {subject.subject_id}/{photo.image_id}")

    print(f"Saved {saved} mouth landmarks.")


if __name__ == "__main__":
    run(Project.open())
