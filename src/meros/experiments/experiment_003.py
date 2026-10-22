"""Experiment 003: spatial spot matching with forced mask-based registration."""

from __future__ import annotations

import csv
import json

import cv2 as cv
import numpy as np

from dataclasses import dataclass
from itertools import combinations
from numpy.typing import NDArray
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from typing import Literal, TypedDict, cast

from meros.domain import TrackSelection, VideoAnnotation
from meros.experiments.common import experiment_output, git_revision
from meros.project import Project


type FrameLabel = Literal["reference", "offset"]

type Image = NDArray[np.uint8]
type Points = NDArray[np.float32]

type SampleKey = tuple[str, FrameLabel]

# hard-set pairs for comparison,
# for example, to test very distant frames
pairs = {
    "h01": ("170", "510"),
}

# Sample selection
FRAME_DELTA = 30  # how many frames forward/backward to look for a comparison frame?
MIN_QUALITY = 3

# Spot detection
MAX_SPOTS = 320  # how many spots to consider per image?
SPOT_THRESHOLD_BLOCK_SIZE = 35
SPOT_THRESHOLD_C = 8

# SIFT alignment
MIN_ALIGNMENT_INLIERS = 12
SIFT_FEATURES = 2500  # how many SIFT features to extract per image?
SIFT_MATCH_RATIO = 0.7  # best SIFT match must be clearly better than the second-best
SIFT_RANSAC_THRESHOLD_PX = 4.0

# Spatial matching
MAX_SPATIAL_DISTANCE_PX = 8.0


class ComparisonRow(TypedDict):
    source: str
    target: str
    same_individual: bool
    source_spots: int
    target_spots: int
    alignment_method: str
    alignment_inliers: int
    spatial_matches: int
    spatial_source_recall: float | None
    spatial_target_recall: float | None
    spatial_repeatability: float | None
    spatial_median_error_px: float | None


@dataclass(frozen=True)
class Sample:
    image: Image
    points: Points
    mask: Image


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


# returns possible spots
def detect_spots(image: Image) -> tuple[Points, Image]:
    if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGRA masked crop")

    height, width = image.shape[:2]

    # erode the mask to avoid edge artifacts; only consider fully valid pixels for spot detection
    valid = cv.erode(
        (image[:, :, 3] == 255).astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    gray = cv.cvtColor(image[:, :, :3], cv.COLOR_BGR2GRAY)
    gray_active = gray[valid]

    # draw the annotated image in BGR, ignoring the alpha channel
    annotated = cast(Image, cv.cvtColor(image, cv.COLOR_BGRA2BGR))

    if gray_active.size == 0:
        return np.empty((0, 2), dtype=np.float32), annotated

    filled = gray.copy()
    filled[~valid] = int(np.median(gray_active))

    enhanced = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filled)

    # adaptative limiar so that spots are detected even in low-contrast regions ("""pattern extraction""")
    binary = cv.adaptiveThreshold(
        enhanced,
        255,  # maxValue
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        SPOT_THRESHOLD_BLOCK_SIZE,  # blockSize
        SPOT_THRESHOLD_C,  # C
    )

    # never let invalid or transparent pixels become candidate spots.
    binary[~valid] = 0

    count, _, stats, centers = cv.connectedComponentsWithStats(binary, connectivity=8)

    found: list[tuple[float, float, int]] = []

    for index in range(1, count):
        left, top, component_width, component_height, area = map(int, stats[index])

        # remove tiny spots and spots that are too large relative to the image size
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

        if not (
            0.35 <= aspect <= 2.8 and 0.25 <= density <= 0.94
        ):  # heuristics: should not be too elongated or sparse
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


def coarse_alignment(source: Sample, target: Sample) -> tuple[Points, int, bool]:
    """Estimate initial geometry using image features, independently of spot descriptors."""

    sift = cv.SIFT.create(nfeatures=SIFT_FEATURES)

    source_gray = cv.cvtColor(source.image, cv.COLOR_BGR2GRAY)
    target_gray = cv.cvtColor(target.image, cv.COLOR_BGR2GRAY)

    source_keys, source_desc = sift.detectAndCompute(source_gray, source.mask)
    target_keys, target_desc = sift.detectAndCompute(target_gray, target.mask)

    if source_desc is None or target_desc is None:
        return source.points.copy(), 0, False

    matcher = cv.BFMatcher(cv.NORM_L2)

    matches = [
        pair[0]
        for pair in matcher.knnMatch(source_desc, target_desc, k=2)
        if len(pair) == 2 and pair[0].distance < SIFT_MATCH_RATIO * pair[1].distance
    ]

    if len(matches) < MIN_ALIGNMENT_INLIERS:
        return source.points.copy(), 0, False

    source_keys_points = np.asarray(
        [source_keys[pair.queryIdx].pt for pair in matches], dtype=np.float32
    )
    target_keys_points = np.asarray(
        [target_keys[pair.trainIdx].pt for pair in matches], dtype=np.float32
    )

    cv.setRNGSeed(0)

    matrix, inliers = cv.estimateAffinePartial2D(
        source_keys_points,
        target_keys_points,
        method=cv.RANSAC,
        ransacReprojThreshold=SIFT_RANSAC_THRESHOLD_PX,
        maxIters=3000,
        confidence=0.99,
    )

    count = int(inliers.sum()) if inliers is not None else 0

    if matrix is None or count < MIN_ALIGNMENT_INLIERS:
        return source.points.copy(), count, False

    transformed = source.points @ matrix[:, :2].T + matrix[:, 2]

    return np.asarray(transformed, dtype=np.float32), count, True


def mask_alignment(source: Sample, target: Sample) -> Points:
    """Register mask bounding boxes without using image texture or spot matches."""
    source_y, source_x = np.nonzero(source.mask)
    target_y, target_x = np.nonzero(target.mask)

    if len(source_x) == 0 or len(target_x) == 0:
        return np.empty((0, 2), dtype=np.float32)

    source_center = np.array(
        [(source_x.min() + source_x.max()) / 2, (source_y.min() + source_y.max()) / 2],
        dtype=np.float32,
    )
    target_center = np.array(
        [(target_x.min() + target_x.max()) / 2, (target_y.min() + target_y.max()) / 2],
        dtype=np.float32,
    )

    source_width = max(float(source_x.max() - source_x.min()), 1.0)
    source_height = max(float(source_y.max() - source_y.min()), 1.0)
    target_width = max(float(target_x.max() - target_x.min()), 1.0)
    target_height = max(float(target_y.max() - target_y.min()), 1.0)

    source_scale = max(source_width, source_height)
    target_scale = max(target_width, target_height)

    return np.asarray(
        (source.points - source_center) * (target_scale / source_scale) + target_center,
        dtype=np.float32,
    )


def spatial_matches(aligned: Points, target: Points) -> tuple[list[tuple[int, int]], float | None]:
    """Assign nearby spots uniquely; allow every source spot to remain unmatched."""
    if len(aligned) == 0 or len(target) == 0:
        return [], None

    distances = np.linalg.norm(aligned[:, None, :] - target[None, :, :], axis=2)

    unmatched = np.full((len(aligned), len(aligned)), MAX_SPATIAL_DISTANCE_PX)

    costs = np.concatenate((distances, unmatched), axis=1)

    source_indices, assigned_columns = linear_sum_assignment(costs)

    pairs = [
        (int(i), int(j))
        for i, j in zip(source_indices, assigned_columns, strict=True)
        if j < len(target) and distances[i, j] <= MAX_SPATIAL_DISTANCE_PX
    ]

    median = float(np.median([distances[i, j] for i, j in pairs])) if pairs else None

    return pairs, median


def compare(source: Sample, target: Sample) -> MatchResult:
    aligned, alignment_inliers, alignment_applied = coarse_alignment(source, target)

    if alignment_applied:
        alignment_method = "sift"
    else:
        alignment_method = "mask_bbox"
        aligned = mask_alignment(source, target)

    if len(aligned) == 0:
        spatial_pairs: list[tuple[int, int]] = []
        spatial_median = None
    else:
        spatial_pairs, spatial_median = spatial_matches(aligned, target.points)

    source_count = len(source.points)
    target_count = len(target.points)

    statistics: ComparisonRow = {
        "source": "",
        "target": "",
        "same_individual": False,
        "source_spots": source_count,
        "target_spots": target_count,
        "alignment_method": alignment_method,
        "alignment_inliers": alignment_inliers,
        "spatial_matches": len(spatial_pairs),
        "spatial_source_recall": len(spatial_pairs) / source_count if source_count else None,
        "spatial_target_recall": len(spatial_pairs) / target_count if target_count else None,
        "spatial_repeatability": (
            len(spatial_pairs) / min(source_count, target_count)
            if source_count and target_count
            else None
        ),
        "spatial_median_error_px": spatial_median,
    }

    return MatchResult(statistics, spatial_pairs)


def draw_matches(source: Sample, target: Sample, result: MatchResult) -> Image:
    source_image = source.image
    target_image = target.image

    height = max(source_image.shape[0], target_image.shape[0])
    width = source_image.shape[1] + target_image.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)

    output[: source_image.shape[0], : source_image.shape[1]] = source_image
    output[: target_image.shape[0], source_image.shape[1] :] = target_image

    for point in source.points:
        cv.circle(output, (round(float(point[0])), round(float(point[1]))), 3, (0, 255, 255), 1)

    for point in target.points:
        cv.circle(
            output,
            (round(float(point[0])) + source_image.shape[1], round(float(point[1]))),
            3,
            (0, 255, 255),
            1,
        )

    for source_index, target_index in result.pairs:
        source_point = (
            int(round(float(source.points[source_index, 0]))),
            int(round(float(source.points[source_index, 1]))),
        )

        target_point = (
            int(round(float(target.points[target_index, 0]))) + source_image.shape[1],
            int(round(float(target.points[target_index, 1]))),
        )

        cv.line(output, source_point, target_point, (0, 220, 0), 1)

    return output


# return a list of all eligible selections for comparison:
# - frontal viewpoint
# - quality >= MIN_QUALITY
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


# return a to-compare frame based on a selection, forward or backward by FRAME_DELTA
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


def run(project: Project, output: Path) -> list[ComparisonRow]:
    selections = eligible_selections(project)

    if not selections:
        raise ValueError("No individual selections have frontal annotations with quality >= 3")

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")

    output.mkdir(parents=True)

    samples: dict[SampleKey, Sample] = {}

    for selection in selections:
        frames: list[tuple[FrameLabel, int]] = []

        if selection.track.video_id not in pairs.keys():
            frames = [("reference", selection.reference_frame)]

            offset = offset_frame(selection)

            if offset is not None:
                frames.append(("offset", offset))
        else:
            reference, offset = pairs[selection.track.video_id]

            frames.append(("reference", int(reference)))
            frames.append(("offset", int(offset)))

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

            p = output / name
            if not cv.imwrite(p, annotated):
                raise OSError(f"Could not write {p}")

            mask = np.asarray((frame[:, :, 3] == 255).astype(np.uint8) * 255, dtype=np.uint8)

            clean = cast(Image, cv.cvtColor(frame, cv.COLOR_BGRA2BGR))

            samples[selection.sample_id, label] = Sample(clean, points, mask)

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

        result = compare(source, target)

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

        p = output / name
        if not cv.imwrite(str(p), figure):
            raise OSError(f"Could not write {p}")

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
        "max_spots": MAX_SPOTS,
        "minimum_alignment_inliers": MIN_ALIGNMENT_INLIERS,
        "maximum_spatial_distance_px": MAX_SPATIAL_DISTANCE_PX,
        "fallback_alignment": "mask_bbox_center_uniform_scale",
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
