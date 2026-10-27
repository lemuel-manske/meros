import cv2 as cv
import matplotlib.pyplot as plt

from meros.project import Project


if __name__ == "__main__":
    project = Project.open()

    for vid in project.metadata.read_videos():
        individuals = vid.individuals

        annotations = vid.matching_annotations(
            viewpoint="frontal",
            min_quality=3,
        )

        for annotation in annotations:
            path = project.media.masked_crop_path(
                vid.video_id,
                annotation.track_id,
                (annotation.start_frame + annotation.end_frame) // 2,
            )

            img = cv.imread(path)
            assert img is not None

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            sift = cv.SIFT.create()

            keypoints, descriptors = sift.detectAndCompute(gray, None)

            r = cv.drawKeypoints(
                gray,
                keypoints,
                gray,
                color=(0, 255, 0),
                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            plt.figure(figsize=(10, 6))
            plt.imshow(cv.cvtColor(r, cv.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
