from dataclasses import replace

import cv2 as cv
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

from meros.domain.storage import BBox
from meros.project import Project

VIDEO_IDS = set()

ONLY_PENDING = True


def select_bboxes(image: np.ndarray, initial: list[BBox]) -> tuple[list[BBox], bool] | None:
    boxes = [box.copy() for box in initial]

    patches = []
    result = None

    figure, axis = plt.subplots(figsize=(15, 9))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.85)

    axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    axis.set_axis_off()

    axis.set_title(
        "Select only Goliath Groupers\n"
        "Drag: add | Right click: remove | C: clear | Backspace: undo\n"
        "Enter: confirm (empty = no meros) | X: exclude ambiguous frame | Q/Esc: exit"
    )

    def refresh() -> None:
        for patch in patches:
            patch.remove()

        patches.clear()

        for x1, y1, x2, y2 in boxes:
            patch = Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2
            )
            axis.add_patch(patch)
            patches.append(patch)

        figure.canvas.draw_idle()

    def on_select(start, end) -> None:
        if any(value is None for value in (start.xdata, start.ydata, end.xdata, end.ydata)):
            return

        height, width = image.shape[:2]
        x1, x2 = sorted((round(start.xdata), round(end.xdata)))
        y1, y2 = sorted((round(start.ydata), round(end.ydata)))
        bbox = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]

        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            boxes.append(bbox)

        selector.clear()
        refresh()

    def on_click(event) -> None:
        if event.button != MouseButton.RIGHT or event.inaxes is not axis:
            return

        matches = [
            i
            for i, (x1, y1, x2, y2) in enumerate(boxes)
            if x1 <= event.xdata <= x2 and y1 <= event.ydata <= y2
        ]

        if matches:
            index = min(
                matches,
                key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
            )
            boxes.pop(index)
            refresh()

    def on_key(event) -> None:
        nonlocal result

        if event.key == "enter":
            result = ([box.copy() for box in boxes], False)
            plt.close(figure)
        elif event.key == "x":
            result = ([], True)
            plt.close(figure)
        elif event.key in ("q", "escape"):
            plt.close(figure)
        elif event.key == "c":
            boxes.clear()
            refresh()
        elif event.key in ("backspace", "ctrl+z") and boxes:
            boxes.pop()
            refresh()

    selector = RectangleSelector(
        axis,
        on_select,
        button=[MouseButton.LEFT],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        props={"facecolor": "yellow", "edgecolor": "yellow", "alpha": 0.2},
    )
    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)

    refresh()
    plt.show(block=True)

    return result


def run(project: Project) -> None:
    for video in project.videos.get_all():
        if video._id in VIDEO_IDS:
            continue

        frames = [
            frame
            for frame in video.frames
            if frame.annotations is not None
            and (not ONLY_PENDING or not frame.annotations.reviewed)
        ]

        print(f"{len(frames)} frames to review")

        for frame in frames:
            annotation = frame.annotations

            assert annotation is not None

            initial = annotation.bboxes if annotation.reviewed else annotation.suggestions

            selected = select_bboxes(frame.im, initial)

            if selected is None:
                break

            boxes, excluded = selected
            project.videos.write_frame_annotation(
                video._id,
                frame,
                replace(annotation, bboxes=boxes, reviewed=True, excluded=excluded),
            )

            print(f"Saved {video._id}/{frame._id}: {len(boxes)} meros, excluded={excluded}")


if __name__ == "__main__":
    run(Project.open())
