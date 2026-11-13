from meros.project import Project

FRAME_STEP = 6
STEPS_EACH_SIDE = 5


def run(project: Project) -> None:
    for video in project.videos.get_all():
        pivots = [frame for frame in video.selected_frames]

        for pivot in pivots:
            indexes = range(
                pivot.frame_index - FRAME_STEP * STEPS_EACH_SIDE,
                pivot.frame_index + FRAME_STEP * STEPS_EACH_SIDE + 1,
                FRAME_STEP,
            )
            frames = [video.frames[index] for index in indexes if 0 <= index < len(video.frames)]

            project.videos.write_frame_neighbors(video, pivot, frames)

            print(
                f"{video.video_id}/{pivot.frame_index}: {len(frames)} frames including pivot",
                flush=True,
            )


if __name__ == "__main__":
    run(Project.open())
