"""Experiment 003: head-spot geometry with affine RANSAC verification."""

from __future__ import annotations

import csv
import json

import cv2 as cv
import numpy as np

from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from itertools import combinations
from scipy.spatial import KDTree

from meros.domain import TrackSelection, VideoAnnotation
from meros.experiments.common import experiment_output, git_revision
from meros.project import Project
from typing import Literal, TypedDict, cast


type FrameLabel = Literal["reference", "offset"]

type Image = NDArray[np.uint8]
type Points = NDArray[np.float32]

type SampleKey = tuple[str, FrameLabel]

MIN_QUALITY = 3
FRAME_DELTA = 30
NEIGHBORS = 6
MAX_SPOTS = 180
MIN_RANSAC_MATCHES = 8


class ComparisonRow(TypedDict):
    source: str
    target: str
    same_individual: bool
    source_spots: int
    target_spots: int
    tentative_matches: int
    ransac_attempted: bool
    inliers: int
    inlier_ratio: float | None
    median_residual_px: float | None
    source_coverage: float
    target_coverage: float


@dataclass(frozen=True)
class Sample:
    image: Image
    points: Points


@dataclass(frozen=True)
class EligibleSelection:
    individual_id: str
    track: TrackSelection
    annotation: VideoAnnotation
    reference_frame: int

    @property
    def sample_id(self) -> str:
        return f"{self.track.selection_id}--frontal-{self.annotation.start_frame}-{self.annotation.end_frame}"


@dataclass(frozen=True)
class MatchResult:
    statistics: ComparisonRow
    pairs: list[tuple[int, int]]
    inlier_mask: list[bool]


def detect_spots(image: Image) -> tuple[Points, Image]:
    if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGRA masked crop")

    height, width = image.shape[:2]

    valid = cv.erode(
        (image[:, :, 3] == 255).astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    gray = cv.cvtColor(image[:, :, :3], cv.COLOR_BGR2GRAY)

    active = gray[valid]

    annotated = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    if active.size == 0:
        return np.empty((0, 2), dtype=np.float32), annotated

    filled = gray.copy()

    filled[~valid] = int(np.median(active))

    enhanced = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filled)

    binary = cv.adaptiveThreshold(
        enhanced,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        35,
        8,
    )

    # Never let invalid or transparent pixels become candidate spots.
    binary[~valid] = 0

    count, _, stats, centers = cv.connectedComponentsWithStats(binary, connectivity=8)

    found: list[tuple[float, float, int]] = []

    for index in range(1, count):
        left, top, component_width, component_height, area = map(int, stats[index])

        if not (5 <= area <= max(10, int(width * height * 0.001))):
            continue

        if component_width < 2 or component_height < 2:
            continue

        if (
            left == 0
            or top == 0
            or left + component_width >= width
            or top + component_height >= height
        ):
            continue

        aspect = component_width / component_height
        density = area / (component_width * component_height)

        if not (0.35 <= aspect <= 2.8 and 0.25 <= density <= 0.94):
            continue

        if not np.all(valid[top : top + component_height, left : left + component_width]):
            continue

        center_x, center_y = centers[index]

        found.append((float(center_x), float(center_y), area))

    found.sort(key=lambda item: (-item[2], item[1], item[0]))

    points = np.asarray(
        [(center_x, center_y) for center_x, center_y, _ in found[:MAX_SPOTS]],
        dtype=np.float32,
    ).reshape(-1, 2)

    for center_x, center_y in points:
        center = (round(float(center_x)), round(float(center_y)))

        cv.circle(annotated, center, 4, (0, 255, 255), 1)

    return points, annotated


def descriptors(points: Points, neighbors: int = NEIGHBORS) -> Points:
    if len(points) < neighbors + 1:
        return np.empty((0, neighbors), dtype=np.float32)

    distances, _ = KDTree(points).query(points, k=neighbors + 1)

    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float32)

    r = neighbor_distances / np.maximum(neighbor_distances[:, -1:], 1e-5)

    return np.asarray(r, dtype=np.float32)


def candidate_matches(source: Points, target: Points) -> list[tuple[int, int]]:
    source_descriptors = descriptors(source)
    target_descriptors = descriptors(target)

    if len(source_descriptors) < 2 or len(target_descriptors) < 2:
        return []

    distances_raw, indices_raw = KDTree(target_descriptors).query(source_descriptors, k=2)

    _, reverse_raw = KDTree(source_descriptors).query(target_descriptors, k=1)

    distances = cast(NDArray[np.float32], distances_raw)
    indices = cast(NDArray[np.int64], indices_raw)
    reverse = cast(NDArray[np.int64], reverse_raw)

    candidates: list[tuple[int, int]] = []

    for source_index in range(len(source_descriptors)):
        target_index = int(indices[source_index, 0])

        if distances[source_index, 0] >= 0.88 * max(distances[source_index, 1], 1e-10):
            continue

        if int(reverse[target_index]) != source_index:
            continue

        candidates.append((source_index, target_index))

    return candidates


def compare(source: Points, target: Points) -> MatchResult:
    pairs = candidate_matches(source, target)

    statistics: ComparisonRow = {
        "source": "",
        "target": "",
        "same_individual": False,
        "source_spots": len(source),
        "target_spots": len(target),
        "tentative_matches": len(pairs),
        "ransac_attempted": False,
        "inliers": 0,
        "inlier_ratio": None,
        "median_residual_px": None,
        "source_coverage": 0.0,
        "target_coverage": 0.0,
    }

    inlier_mask = [False] * len(pairs)

    if len(pairs) < MIN_RANSAC_MATCHES:
        return MatchResult(statistics, pairs, inlier_mask)

    source_points = np.asarray([source[i] for i, _ in pairs], dtype=np.float32)
    target_points = np.asarray([target[j] for _, j in pairs], dtype=np.float32)

    cv.setRNGSeed(0)

    matrix, mask = cv.estimateAffine2D(
        source_points,
        target_points,
        method=cv.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=3000,
        confidence=0.99,
        refineIters=10,
    )

    statistics["ransac_attempted"] = True

    if matrix is None or mask is None or np.linalg.det(matrix[:, :2]) <= 0:
        return MatchResult(statistics, pairs, inlier_mask)

    accepted = mask.ravel().astype(bool)

    residuals = np.linalg.norm(
        source_points @ matrix[:, :2].T + matrix[:, 2] - target_points,
        axis=1,
    )

    count = int(accepted.sum())

    statistics["inliers"] = count
    statistics["inlier_ratio"] = count / len(pairs)
    statistics["median_residual_px"] = float(np.median(residuals[accepted])) if count else None
    statistics["source_coverage"] = count / max(1, len(source))
    statistics["target_coverage"] = count / max(1, len(target))

    return MatchResult(statistics, pairs, accepted.tolist())


def draw_matches(source: Sample, target: Sample, result: MatchResult) -> Image:
    source_image = source.image
    target_image = target.image

    height = max(source_image.shape[0], target_image.shape[0])
    width = source_image.shape[1] + target_image.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)

    output[: source_image.shape[0], : source_image.shape[1]] = source_image
    output[: target_image.shape[0], source_image.shape[1] :] = target_image

    for (source_index, target_index), accepted in zip(
        result.pairs, result.inlier_mask, strict=True
    ):
        source_point = (
            int(round(float(source.points[source_index, 0]))),
            int(round(float(source.points[source_index, 1]))),
        )

        target_point = (
            int(round(float(target.points[target_index, 0]))) + source_image.shape[1],
            int(round(float(target.points[target_index, 1]))),
        )

        color = (0, 220, 0) if accepted else (40, 80, 220)

        cv.line(output, source_point, target_point, color, 1)

    return output


def eligible_selections(project: Project) -> list[EligibleSelection]:
    videos = {video.video_id: video for video in project.metadata.read_videos()}

    eligible: list[EligibleSelection] = []

    for individual in project.metadata.read_individuals():
        for track in individual.tracks:
            video = videos.get(track.video_id)

            if video is None:
                raise ValueError(f"Video metadata missing for {track.video_id}")

            annotations = video.matching_annotations(
                viewpoint="frontal",
                min_quality=MIN_QUALITY,
                track_id=track.track_id,
            )

            for annotation in annotations:
                start = max(track.start_frame, annotation.start_frame)
                end = min(track.end_frame, annotation.end_frame)

                if start > end:
                    continue

                reference = track.reference_frame

                if not start <= reference <= end:
                    reference = (start + end) // 2

                eligible.append(
                    EligibleSelection(
                        individual_id=individual.individual_id,
                        track=track,
                        annotation=annotation,
                        reference_frame=reference,
                    )
                )

    return eligible


def offset_frame(selection: EligibleSelection) -> int | None:
    start = max(selection.track.start_frame, selection.annotation.start_frame)
    end = min(selection.track.end_frame, selection.annotation.end_frame)

    forward = selection.reference_frame + FRAME_DELTA

    if forward <= end:
        return forward

    backward = selection.reference_frame - FRAME_DELTA

    if backward >= start:
        return backward

    return None


def save_image(path: Path, image: Image) -> None:
    if not cv.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def run(project: Project, output: Path) -> list[ComparisonRow]:
    selections = eligible_selections(project)

    if not selections:
        raise ValueError(
            "No individual selections have frontal annotations with quality >= 3"
        )

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")

    output.mkdir(parents=True)

    samples: dict[SampleKey, Sample] = {}

    for selection in selections:
        frames: list[tuple[FrameLabel, int]] = [
            ("reference", selection.reference_frame)
        ]

        offset = offset_frame(selection)

        if offset is not None:
            frames.append(("offset", offset))

        for label, frame_index in frames:
            frame = cast(
                Image,
                project.media.read_masked_crop(
                    selection.track.video_id,
                    selection.track.track_id,
                    frame_index,
                ),
            )

            points, annotated = detect_spots(frame)

            name = f"{selection.sample_id}--{label}--spots.png"

            save_image(output / name, annotated)

            samples[selection.sample_id, label] = Sample(annotated, points)

    comparisons: list[tuple[EligibleSelection, FrameLabel, EligibleSelection, FrameLabel]] = []

    for selection in selections:
        if (selection.sample_id, "offset") in samples:
            comparisons.append((selection, "reference", selection, "offset"))

    for source, target in combinations(selections, 2):
        comparisons.append((source, "reference", target, "reference"))

    if not comparisons:
        raise ValueError("No eligible comparisons; need two selections or an offset frame")

    rows: list[ComparisonRow] = []

    for source_selection, source_label, target_selection, target_label in comparisons:
        source_key = (source_selection.sample_id, source_label)
        target_key = (target_selection.sample_id, target_label)

        source = samples[source_key]
        target = samples[target_key]

        result = compare(source.points, target.points)

        result.statistics["source"] = "/".join(source_key)
        result.statistics["target"] = "/".join(target_key)
        result.statistics["same_individual"] = (
            source_selection.individual_id == target_selection.individual_id
        )

        figure = draw_matches(source, target, result)

        name = (
            f"{source_selection.sample_id}--{source_label}--vs--"
            f"{target_selection.sample_id}--{target_label}.png"
        )

        save_image(output / name, figure)

        rows.append(result.statistics)

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ComparisonRow.__annotations__))

        writer.writeheader()

        writer.writerows(rows)

    provenance: dict[str, object] = {
        "source_commit": git_revision(),
        "viewpoint": "frontal",
        "minimum_quality": MIN_QUALITY,
        "frame_delta": FRAME_DELTA,
        "selections": [
            {
                "individual_id": selection.individual_id,
                "selection_id": selection.track.selection_id,
                "annotation_start": selection.annotation.start_frame,
                "annotation_end": selection.annotation.end_frame,
                "reference_frame": selection.reference_frame,
            }
            for selection in selections
        ],
    }

    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"Wrote {len(rows)} comparisons to {output}")

    return rows


def main() -> None:
    project = Project.open()

    output = experiment_output("003")

    run(project, output)


if __name__ == "__main__":
    main()
