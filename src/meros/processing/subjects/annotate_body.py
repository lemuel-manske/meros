import cv2 as cv
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np

from meros.domain.storage import SubjectMark
from meros.project import Project

SUBJECT_ID = "joao"
FRAME_ID = None


def annotate_body(
    crop: np.ndarray, initial: list[SubjectMark], title: str
) -> list[SubjectMark] | None:
    marks = [{"polygon": [point.copy() for point in mark["polygon"]]} for mark in initial]
    points: list[list[int]] = []

    result = None

    figure, axis = plt.subplots(figsize=(13, 8))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.85)

    display = (
        cv.cvtColor(crop, cv.COLOR_BGRA2RGBA)
        if crop.shape[2] == 4
        else cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    )

    axis.imshow(display)
    axis.set_axis_off()
    axis.set_title(
        f"{title}\nLeft click: add polygon vertex | Enter: finish polygon | "
        "Backspace: undo vertex | U: undo mark | C: clear | S: save | Esc/Q: skip"
    )
    artists = []

    def refresh() -> None:
        for artist in artists:
            artist.remove()

        artists.clear()

        for mark in marks:
            polygon = np.asarray(mark["polygon"])

            artists.append(
                axis.add_patch(
                    Polygon(polygon, closed=True, fill=False, edgecolor="yellow", linewidth=2)
                )
            )

        if points:
            artists.append(axis.scatter(*np.asarray(points).T, s=32, color="red"))

        figure.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return

        points.append([round(event.xdata), round(event.ydata)])
        refresh()

    def on_key(event) -> None:
        nonlocal result

        if event.key == "enter":
            if len(points) >= 3:
                marks.append({"polygon": points.copy()})
                points.clear()
                refresh()

        elif event.key == "s":
            if len(points) >= 3:
                marks.append({"polygon": points.copy()})
                points.clear()

            result = marks
            plt.close(figure)

        elif event.key in ("escape", "q"):
            plt.close(figure)

        elif event.key in ("backspace", "ctrl+z") and points:
            points.pop()
            refresh()

        elif event.key == "u" and marks:
            marks.pop()
            refresh()

        elif event.key == "c":
            marks.clear()
            points.clear()
            refresh()

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)

    refresh()
    plt.show(block=True)

    return result


def run(project: Project) -> None:
    for subject in project.subjects.get_all():
        if SUBJECT_ID is not None and subject.subject_id != SUBJECT_ID:
            continue

        for image in subject.images:
            if FRAME_ID is not None and image.image_id != FRAME_ID:
                continue

            title = f"{subject.subject_id} / image {image.image_id}"

            selected = annotate_body(image.crop, image.body, title)

            if selected is None:
                continue

            project.subjects.write_subject_body(subject.subject_id, image.image_id, selected)
            print(f"Saved {len(selected)} body marks for {title}")


if __name__ == "__main__":
    run(Project.open())
