from meros.domain.storage import Subject, SubjectImage
from meros.project import Project

VIDEO_ID = "GX010349"

CANDIDATE_ID = "joao"


def run(project: Project):
    for video in project.videos.get_all():
        if VIDEO_ID is not None and video.video_id != VIDEO_ID:
            continue

        pivots = [frame for frame in video.selected_frames]

        subject_imaages = []

        for pivot in pivots:
            im = SubjectImage(
                subject_id=CANDIDATE_ID,
                capture_date=pivot.capture_date,
                image_id=pivot.frame_index,
                path=pivot.path,
                _load_im=lambda pivot=pivot: pivot.im,
                _load_crop=lambda pivot=pivot: pivot.im,
            )

            subject_imaages.append(im)

        candidate = Subject(subject_id=CANDIDATE_ID, images=subject_imaages)

        project.subjects.write_subject(candidate, source_video_id=video.video_id)

        print(f"Written candidate {CANDIDATE_ID} with {len(subject_imaages)} images to project.")


if __name__ == "__main__":
    run(Project.open())
