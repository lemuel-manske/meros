from pathlib import Path
from tempfile import TemporaryDirectory

import cv2 as cv
import numpy as np
import torch

from meros.external import build_sam2_predictor
from meros.project import Project
from meros.utils import configure_torch

SUBJECT_ID = "sandra"
FRAME_ID = None


def run(project: Project) -> None:
    configure_torch()

    predictor = build_sam2_predictor(project.sam2_model)

    with torch.inference_mode():
        for subject in project.subjects.get_all():
            if SUBJECT_ID is not None and subject.subject_id != SUBJECT_ID:
                continue

            for photo in subject.images:
                if FRAME_ID is not None and photo.image_id != FRAME_ID:
                    continue

                name = f"{subject.subject_id}/{photo.image_id}"

                assert len(photo.bbox) == 4, f"{name}: missing bbox"

                x1, y1, x2, y2 = photo.bbox
                height, width = photo.im.shape[:2]

                assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, name

                with TemporaryDirectory() as directory:
                    frame = Path(directory) / "00000000.jpg"
                    frame.symlink_to(Path(photo.path).resolve())

                    state = predictor.init_state(directory)

                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=1,
                        box=np.asarray(photo.bbox, dtype=np.float32),
                    )

                    mask = next(predictor.propagate_in_video(state)).masks["1"]

                    del state

                crop_mask = mask[y1:y2, x1:x2]
                assert crop_mask.any(), f"{name}: empty SAM2 crop"

                crop = cv.cvtColor(photo.im[y1:y2, x1:x2], cv.COLOR_BGR2BGRA)
                crop[~crop_mask] = 0

                project.subjects.write_subject_crop(subject.subject_id, photo.image_id, crop)

                print(f"Saved crop for {name}", flush=True)


if __name__ == "__main__":
    run(Project.open())
