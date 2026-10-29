"""Show verified and proposed spot correspondences for Luana side by side."""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meros.processing.compare_joint_spot_matching import match
from meros.processing.compare_spot_geometry import load
from meros.project import Project


SUBJECT_ID = "luana"
FIRST_IMAGE_ID = 0
SECOND_IMAGE_ID = 1


def run(project: Project) -> None:
    samples = load(project)
    first = next(
        sample for sample in samples
        if (sample.subject_id, sample.image_id) == (SUBJECT_ID, FIRST_IMAGE_ID)
    )
    second = next(
        sample for sample in samples
        if (sample.subject_id, sample.image_id) == (SUBJECT_ID, SECOND_IMAGE_ID)
    )

    predicted_indices = match(first.normalized, second.normalized)
    predicted = [
        (first.spots[i], second.spots[j]) for i, j in predicted_indices
    ]

    annotated = project.metadata.read_subject_spot_correspondences(
        SUBJECT_ID, FIRST_IMAGE_ID, SECOND_IMAGE_ID
    )
    verified = [
        (pair["first"], pair["second"]) for pair in annotated
    ]

    images = [
        project.media.read_subject_image(SUBJECT_ID, FIRST_IMAGE_ID),
        project.media.read_subject_image(SUBJECT_ID, SECOND_IMAGE_ID),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(18, 12))

    for row, (title, pairs) in enumerate(
        (("Verified", verified), ("Proposed", predicted))
    ):
        for col, (axis, image, image_id) in enumerate(
            zip(axes[row], images, (FIRST_IMAGE_ID, SECOND_IMAGE_ID), strict=True)
        ):
            axis.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            axis.set_title(f"{title}: {SUBJECT_ID}/{image_id}")
            axis.set_axis_off()

            for number, pair in enumerate(pairs, start=1):
                x, y = pair[col]
                axis.scatter(
                    [x], [y], s=35, facecolors="none",
                    edgecolors="yellow", linewidths=1.5,
                )
                axis.annotate(
                    str(number), (x, y), xytext=(5, -5),
                    textcoords="offset points",
                    color="yellow", fontweight="bold", fontsize=9,
                )

    figure.tight_layout()
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    run(Project.open())
