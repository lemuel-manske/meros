"""Compare ordinary SIFT and morphology-filtered SIFT on frontal head crops."""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.ml.spot_sift import extract_spot_sift
from meros.project import Project


if __name__ == "__main__":
    project = Project.open()

    for vid in project.metadata.read_videos():
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

            image = cv.imread(path, cv.IMREAD_UNCHANGED)

            if image is None:
                raise FileNotFoundError(path)

            if image.ndim == 3 and image.shape[2] == 4:
                mask = (image[:, :, 3] == 255).astype(np.uint8) * 255
                bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
            else:
                mask = None
                bgr = image

            gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
            sift = cv.SIFT.create(nfeatures=2500)
            baseline, _ = sift.detectAndCompute(gray, mask)
            result = extract_spot_sift(image, mask)

            before = cv.drawKeypoints(
                bgr,
                baseline,
                None,  # pyright: ignore
                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            after = cv.drawKeypoints(
                bgr,
                result.keypoints,
                None,  # pyright: ignore
                flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            for x, y in result.spots:
                cv.circle(after, (round(float(x)), round(float(y))), 3, (0, 255, 255), 1)

            figure, axes = plt.subplots(1, 2, figsize=(15, 6))

            axes[0].imshow(cv.cvtColor(before, cv.COLOR_BGR2RGB))
            axes[0].set_title(f"Original SIFT: {len(baseline)} keypoints")

            axes[1].imshow(cv.cvtColor(after, cv.COLOR_BGR2RGB))
            axes[1].set_title(
                f"Spot-filtered SIFT: {len(result.keypoints)} keypoints, "
                f"{len(result.spots)} candidates"
            )

            for axis in axes:
                axis.axis("off")

            figure.suptitle(f"{vid.video_id} / {annotation.track_id}")
            figure.tight_layout()
            plt.show()
            plt.close(figure)
