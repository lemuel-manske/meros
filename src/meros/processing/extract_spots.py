from __future__ import annotations

import cv2 as cv
import matplotlib.pyplot as plt

from meros.project import Project


MIN_QUALITY = 3


def select_spots(image, initial: list[list[int]], title: str) -> list[list[int]] | None:
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

        x = round(event.xdata)
        y = round(event.ydata)

        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            return

        if event.button == 1:
            points.append([x, y])
        elif event.button == 3 and points:
            nearest = min(
                range(len(points)),
                key=lambda i: (points[i][0] - x) ** 2 + (points[i][1] - y) ** 2,
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
    for video in project.metadata.read_videos():
        for annotation in video.matching_annotations(
            viewpoint="frontal",
            min_quality=MIN_QUALITY,
        ):
            frame = (annotation.start_frame + annotation.end_frame) // 2
            path = project.media.masked_crop_path(video.video_id, annotation.track_id, frame)

            image = cv.imread(str(path), cv.IMREAD_UNCHANGED)

            if image is None:
                print(f"Missing image, skipping: {path}")
                continue

            if image.ndim == 3 and image.shape[2] == 4:
                mask = image[:, :, 3]
                bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
            else:
                mask = None
                bgr = image if image.ndim == 3 else cv.cvtColor(image, cv.COLOR_GRAY2BGR)

            existing = project.metadata.read_spots(video.video_id, annotation.track_id, frame)

            title = (
                f"{video.video_id} / track {annotation.track_id} / frame {frame} "
                f"({len(existing)} saved)\n"
                "Left click: add | Right click: remove nearest | Backspace: undo | "
                "Enter/S: save | Esc/Q: skip"
            )
            selected = select_spots(bgr, existing, title)

            if selected is None:
                continue

            if mask is not None:
                invalid = [
                    point
                    for point in selected
                    if mask[point[1], point[0]] == 0
                ]

                if invalid:
                    print(f"Warning: {len(invalid)} points outside fish mask; not saved.")
                    continue

            project.metadata.write_spots(
                video.video_id, annotation.track_id, frame, selected
            )
            print(f"Saved {len(selected)} spot centers for {video.video_id}/{annotation.track_id}/{frame}")


if __name__ == "__main__":
    run(Project.open())
