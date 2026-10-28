"""Draw evaluation and exclusion polygons on annotated head crops."""

from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon

from meros.project import Project


# Edit to limit which annotated crops are reviewed.
VIDEO_ID: str | None = None
TRACK_ID: str | None = None


def edit_regions(image: np.ndarray, regions: dict, name: str) -> dict | None:
    include = [[list(point) for point in polygon] for polygon in regions["include"]]
    exclude = [[list(point) for point in polygon] for polygon in regions["exclude"]]
    complete = bool(regions["complete"])
    kind = "include"
    pending: list[list[int]] = []
    result: dict | None = None

    figure, axis = plt.subplots(figsize=(13, 8))

    def redraw() -> None:
        axis.clear()
        axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        for polygons, color in ((include, "lime"), (exclude, "red")):
            for polygon in polygons:
                axis.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor=color, linewidth=2))

        if pending:
            coords = np.asarray(pending)
            axis.plot(coords[:, 0], coords[:, 1], "yo-")

        axis.set_title(
            f"{name} | active={kind} | complete={complete}\n"
            "Left click: vertex | Enter: close polygon | I: include | E: exclude | "
            "Backspace: undo | C: toggle complete | S: save | Q: skip"
        )
        axis.axis("off")
        figure.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not axis or event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = round(event.xdata), round(event.ydata)

        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            pending.append([x, y])
            redraw()

    def on_key(event) -> None:
        nonlocal kind, complete, result

        if event.key == "i":
            kind = "include"
        elif event.key == "e":
            kind = "exclude"
        elif event.key == "enter":
            if len(pending) >= 3:
                (include if kind == "include" else exclude).append(pending.copy())
                pending.clear()
        elif event.key == "backspace":
            if pending:
                pending.pop()
            elif (include if kind == "include" else exclude):
                (include if kind == "include" else exclude).pop()
        elif event.key == "c":
            complete = not complete
        elif event.key == "s":
            if pending:
                print("Close or discard the unfinished polygon before saving.")
                return

            if not include:
                print("At least one inclusion polygon is required.")
                return

            result = {"include": include, "exclude": exclude, "complete": complete}
            plt.close(figure)
        elif event.key in ("q", "escape"):
            plt.close(figure)

        redraw()

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show(block=True)

    return result


def run(project: Project) -> None:
    for video_id, track_id, frame in project.metadata.list_spot_annotations():
        if VIDEO_ID is not None and video_id != VIDEO_ID:
            continue

        if TRACK_ID is not None and track_id != TRACK_ID:
            continue

        image = project.media.read_masked_crop(video_id, track_id, frame)
        bgr = (
            cv.cvtColor(image, cv.COLOR_BGRA2BGR)
            if image.ndim == 3 and image.shape[2] == 4
            else cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            if image.ndim == 2
            else image
        )
        saved = project.metadata.read_spot_regions(video_id, track_id, frame)
        edited = edit_regions(bgr, saved, f"{video_id}/{track_id}/{frame}")

        if edited is not None:
            project.metadata.write_spot_regions(video_id, track_id, frame, edited)
            print(f"Saved evaluation regions for {video_id}/{track_id}/{frame}")


if __name__ == "__main__":
    run(Project.open())
