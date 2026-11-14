import cv2
import numpy as np
from PIL import Image

from meros.external.gloper import build_gloper_predictor
from meros.project import Project

SCRIPT = {
    "athila": [0],
    "jose": [0],
    "maira": [0],
    "rafael": [2, 3],
    "rogerio": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "sandra": [35418, 121196, 141534],
    "joao": [92612, 143488, 143557],
    "marco": [144095],
    "kaue": [4796, 62961, 122332],
}

KERNELS = [256]


def clahe(
    image: Image.Image,
    *,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    _l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )

    _l = clahe.apply(_l)

    enhanced = cv2.merge((_l, a, b))  # type: ignore
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)


def run(project: Project) -> None:
    predictor = build_gloper_predictor()

    frames = {}

    for subject in project.subjects.get_all():
        if subject._id not in SCRIPT:
            continue

        for photo in subject.images:
            if photo._id not in SCRIPT[subject._id]:
                continue

            frames[(subject._id, photo._id)] = photo.crop_im

    for (subject_id, frame_id), frame in frames.items():
        clahed = clahe(Image.fromarray(frame))

        images = {
            "clahed": clahed,
        }

        subject_images = []

        for image in project.subjects.get(subject_id).images:
            _bin = image.binary_im

            if _bin is None:
                continue

            subject_images.append((image._id, _bin))

        exists = any(frame_id == image_id for image_id, _ in subject_images)

        if exists:
            continue

        for _, image in images.items():
            result = predictor.predict(image, kernels=KERNELS)

            for _, mask in result.items():
                arr = mask.astype(np.uint8) * 255

                subject = project.subjects.get(subject_id)

                assert subject is not None, f"Subject {subject_id} not found"

                subject_image = subject.get_image(frame_id)

                assert subject_image is not None, (
                    f"Image {frame_id} not found for subject {subject_id}"
                )

                subject_image._load_binary_im = lambda: arr

                project.subjects.write(subject)


if __name__ == "__main__":
    run(Project.open())
