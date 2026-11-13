import cv2 as cv
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from meros.project import Project

VIDEO_ID = "GX030001"

CONFIDENCE_THRESHOLD = 0.25
FRAME_STEP = 6
ONLY_SELECTED = True


def run(project: Project) -> None:
    video = project.videos.get(VIDEO_ID)
    selection = {
        frame.frame_index: frame.selected_detections
        for frame in video.frames
        if frame.selected_detections
    }

    frames = [
        frame
        for frame in video.frames
        if any(
            item.confidence >= CONFIDENCE_THRESHOLD
            for item in (frame.selected_detections if ONLY_SELECTED else frame.detections)
        )
    ]

    assert frames, f"No detections with confidence >= {CONFIDENCE_THRESHOLD} in {VIDEO_ID}"

    index = 0

    figure, axis = plt.subplots(figsize=(15, 9))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.88)

    previous = Button(figure.add_axes((0.36, 0.02, 0.12, 0.05)), f"Previous {FRAME_STEP}")
    following = Button(figure.add_axes((0.52, 0.02, 0.12, 0.05)), f"Next {FRAME_STEP}")

    def refresh() -> None:
        frame = frames[index]

        axis.clear()
        axis.imshow(cv.cvtColor(frame.im, cv.COLOR_BGR2RGB))
        axis.set_axis_off()

        detections = frame.selected_detections if ONLY_SELECTED else frame.detections

        for item in detections:
            if item.confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = item.bbox
            axis.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2)
            )
            axis.text(
                x1,
                y1,
                f"{item.confidence:.1%}",
                color="black",
                backgroundcolor="yellow",
                verticalalignment="top",
            )

        axis.set_title(
            f"{video.video_id} | Frame {frame.frame_index} | {frame.timestamp:.2f}s | "
            f"{index + 1}/{len(frames)} selected frames\n"
            f"Left/Right: -/+{FRAME_STEP} frames | Home/End: first/last | "
            f"{'X: discard | ' if ONLY_SELECTED else ''}Q/Esc: exit"
        )

        figure.canvas.draw_idle()

    def move(step: int) -> None:
        nonlocal index

        index = max(0, min(len(frames) - 1, index + step))
        refresh()

    def on_key(event) -> None:
        if event.key in ("right", "n", " "):
            move(FRAME_STEP)
        elif event.key in ("left", "p"):
            move(-FRAME_STEP)
        elif event.key == "home":
            move(-len(frames))
        elif event.key == "end":
            move(len(frames))
        elif event.key == "x" and ONLY_SELECTED:
            removed = frames.pop(index)
            selection.pop(removed.frame_index)
            project.videos.write_video_selection(video, selection)
            print(f"Discarded {video.video_id}/{removed.frame_index}", flush=True)

            if not frames:
                plt.close(figure)
            else:
                move(0)
        elif event.key in ("q", "escape"):
            plt.close(figure)

    previous.on_clicked(lambda _: move(-FRAME_STEP))
    following.on_clicked(lambda _: move(FRAME_STEP))

    figure.canvas.mpl_connect("key_press_event", on_key)

    refresh()
    plt.show(block=True)


if __name__ == "__main__":
    run(Project.open())
