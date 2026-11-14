import cv2 as cv
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

from meros.domain.storage import BBox
from meros.project import Project

SUBJECT_ID = "sandra"
FRAME_ID =  None


def select_bbox(image: np.ndarray, initial: BBox, title: str) -> BBox | None:
    selected = initial.copy()

    result = None

    figure, axis = plt.subplots(figsize=(15, 9))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88)

    axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    axis.set_axis_off()
    axis.set_title(f"{title}\nDrag: draw bbox | Enter: save | C: clear | Esc/Q: skip")

    patches = []

    def refresh() -> None:
        for patch in patches:
            patch.remove()

        patches.clear()

        if selected:
            x1, y1, x2, y2 = selected

            patches.append(
                axis.add_patch(
                    Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2
                    )
                )
            )

        figure.canvas.draw_idle()

    def on_select(start, end) -> None:
        if any(value is None for value in (start.xdata, start.ydata, end.xdata, end.ydata)):
            return

        height, width = image.shape[:2]

        x1, x2 = sorted((round(start.xdata), round(end.xdata)))
        y1, y2 = sorted((round(start.ydata), round(end.ydata)))

        box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]

        if box[2] > box[0] and box[3] > box[1]:
            selected[:] = box

        selector.clear()
        refresh()

    def on_key(event) -> None:
        nonlocal result

        if event.key == "enter":
            result = selected.copy()
            plt.close(figure)
        elif event.key == "c":
            selected.clear()
            refresh()
        elif event.key in ("escape", "q"):
            plt.close(figure)

    selector = RectangleSelector(
        axis,
        on_select,
        button=[MouseButton.LEFT],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        props={"facecolor": "yellow", "edgecolor": "yellow", "alpha": 0.2},
    )

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

            bbox = select_bbox(image.im, image.bbox, title)

            if bbox is None:
                continue

            project.subjects.write_subject_bbox(subject.subject_id, image.image_id, bbox)

            print(f"Saved bbox for {title}: {bbox}")


if __name__ == "__main__":
    run(Project.open())
