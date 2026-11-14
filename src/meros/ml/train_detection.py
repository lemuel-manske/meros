from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from meros.domain.storage import VideoFrame
from meros.project import Project
from meros.utils import configure_torch, write_image

MODEL = Path("external/yolo/fishial-detector.pt")

OUTPUT_ROOT = Path("data/ml/detection")

TRAIN_RECORDINGS = {"GX10001", "GX10002", "GX10349"}
VAL_RECORDINGS = {"GX10350", "GX10351", "GX10352"}

IMAGE_SIZE = 640
DEVICE = 0
EPOCHS = 20
PATIENCE = 8
BATCH_SIZE = 4
FREEZE_LAYERS = 10
LEARNING_RATE = 0.0001
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5


def prepare_dataset(
    project: Project, directory: Path
) -> tuple[Path, list[tuple[VideoFrame, Path]]]:
    splits = {"train": [], "val": []}
    pending = 0

    for video in project.videos.get_all():
        for frame in video.frames:
            if not frame.selected:
                continue

            if not frame.reviewed:
                pending += 1
                continue

            if frame.excluded:
                continue

            if video._id in TRAIN_RECORDINGS:
                split = "train"
            elif video._id in VAL_RECORDINGS:
                split = "val"
            else:
                continue

            splits[split].append((video._id, frame))

    print(f"Ignoring {pending} unreviewed frames", flush=True)

    validation = []

    for split, frames in splits.items():
        positives = sum(bool(frame.bboxes) for _, frame in frames)

        assert positives > 0, f"No positive frames in {split}"

        print(
            f"{split}: {len(frames)} frames, "
            f"{positives} positive, {len(frames) - positives} negative"
        )

        images = directory / "images" / split
        labels = directory / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)

        for video_id, frame in frames:
            image = frame.im
            height, width = image.shape[:2]
            name = f"{video_id}_{frame._id:08d}"

            image_path = images / f"{name}.jpg"
            label_path = labels / f"{name}.txt"

            write_image(image_path, image)

            lines = []

            for bbox in frame.bboxes:
                x1, y1, x2, y2 = bbox.bbox
                assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height

                values = (
                    (x1 + x2) / (2 * width),
                    (y1 + y2) / (2 * height),
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                )
                lines.append("0 " + " ".join(f"{v:.8f}" for v in values))

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

            if split == "val":
                validation.append((frame, image_path))

    dataset = directory / "dataset.yaml"
    dataset.write_text(
        f"path: {directory.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: mero\n"
    )

    return dataset, validation


def evaluate(
    model,
    dataset: Path,
    validation: list[tuple[VideoFrame, Path]],
    output: Path,
    name: str,
) -> dict:
    from ultralytics.utils.metrics import ConfusionMatrix

    metrics = model.val(
        data=str(dataset),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        plots=False,
        project=str(output),
        name=name,
        exist_ok=True,
        verbose=False,
    )

    counter = ConfusionMatrix(names={0: "mero"})

    for frame, image_path in validation:
        boxes = model.predict(
            str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )[0].boxes.cpu()

        counter.process_batch(
            {"bboxes": boxes.xyxy, "cls": boxes.cls, "conf": boxes.conf},
            {
                "bboxes": torch.tensor(
                    [bbox.bbox for bbox in frame.bboxes],
                    dtype=torch.float32,
                ).reshape(-1, 4),
                "cls": torch.zeros(len(frame.bboxes)),
            },
            conf=CONFIDENCE,
            iou_thres=IOU_THRESHOLD,
        )

    tp = int(counter.matrix[0, 0])
    fp = int(counter.matrix[0, 1])
    fn = int(counter.matrix[1, 0])

    values = {
        "frames": len(validation),
        "confidence": CONFIDENCE,
        "iou": IOU_THRESHOLD,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    print(f"{name}: {values}", flush=True)
    return values


def output_dir() -> Path:
    output = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    return output


def run(project: Project) -> None:
    from ultralytics.models.yolo.detect import DetectionTrainer

    from meros.models.yolo import YOLO

    configure_torch()

    assert not TRAIN_RECORDINGS & VAL_RECORDINGS

    output = output_dir()

    with TemporaryDirectory() as directory:
        dataset, validation = prepare_dataset(project, Path(directory))

        baseline = YOLO(MODEL)
        baseline_eval = evaluate(baseline, dataset, validation, output, "baseline")

        model = YOLO(MODEL)
        model.train(
            data=str(dataset),
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            nbs=BATCH_SIZE,
            device=DEVICE,
            amp=False,
            workers=0,
            freeze=FREEZE_LAYERS,
            optimizer="AdamW",
            lr0=LEARNING_RATE,
            warmup_epochs=1,
            mosaic=0,
            scale=0.2,
            translate=0.05,
            hsv_h=0.01,
            hsv_s=0.2,
            hsv_v=0.2,
            plots=False,
            project=str(output.parent),
            name=output.name,
            exist_ok=True,
            seed=0,
        )

        assert isinstance(model.trainer, DetectionTrainer)

        finetuned = YOLO(model.trainer.best)
        finetuned_eval = evaluate(finetuned, dataset, validation, output, "finetuned")

        print(f"Baseline: {baseline_eval}", flush=True)
        print(f"Finetuned: {finetuned_eval}", flush=True)


if __name__ == "__main__":
    run(Project.open())
