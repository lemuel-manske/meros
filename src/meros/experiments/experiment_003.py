"""Experiment 003: head-spot geometry with affine RANSAC verification."""

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
from scipy.spatial import KDTree
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

# Geometric descriptors
MAX_POSITION_DELTA = 0.22
MIN_RANSAC_MATCHES = 6  # how many tentative matches are required to attempt RANSAC?
NEIGHBORS = 6  # how many nearest neighbors to consider for each spot?
SPOT_MATCH_RATIO = 0.88

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
    tentative_matches: int
    ransac_attempted: bool
    inliers: int
    inlier_ratio: float | None
    median_residual_px: float | None
    source_coverage: float
    target_coverage: float
    alignment_inliers: int
    alignment_applied: bool
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
    inlier_mask: list[bool]
    spatial_pairs: list[tuple[int, int]]


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

        if not (0.35 <= aspect <= 2.8 and 0.25 <= density <= 0.94):  # heuristics: should not be too elongated or sparse
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
    k = neighbors + 1  # skip the point itself

    if len(points) < k:
        return np.empty((0, neighbors), dtype=np.float32)

    distances, _ = KDTree(points).query(points, k)

    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float32)

    r = neighbor_distances / np.maximum(neighbor_distances[:, -1:], 1e-5)  # normalize by the distance to the farthest neighbor

    return np.asarray(r, dtype=np.float32)


# what if fish is more close or farther?
def normalized_positions(points: Points) -> Points:
    if len(points) == 0:
        return points.copy()

    low = points.min(axis=0)
    high = points.max(axis=0)

    extent = np.maximum(high - low, 1.0)

    return np.asarray((points - low) / extent, dtype=np.float32)


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
        confidence=0.99
    )

    count = int(inliers.sum()) if inliers is not None else 0

    if matrix is None or count < MIN_ALIGNMENT_INLIERS:
        return source.points.copy(), count, False

    transformed = source.points @ matrix[:, :2].T + matrix[:, 2]

    return np.asarray(transformed, dtype=np.float32), count, True


def candidate_matches(
    source: Points, target: Points, source_aligned: Points | None = None
) -> list[tuple[int, int]]:
    source_descriptors = descriptors(source)
    target_descriptors = descriptors(target)

    if len(source_descriptors) < 2 or len(target_descriptors) < 2:
        return []

    if source_aligned is None:
        source_position = normalized_positions(source)
        target_position = normalized_positions(target)
    else:
        source_position = source_aligned
        target_position = target

    scale = np.maximum(np.ptp(target_position, axis=0), 1.0)

    source_position = source_position / scale
    target_position = target_position / scale

    position_distances = np.linalg.norm(
        source_position[:, None, :] - target_position[None, :, :],
        axis=2,
    )

    descriptor_distances = np.linalg.norm(
        source_descriptors[:, None, :] - target_descriptors[None, :, :],
        axis=2,
    )

    descriptor_distances[position_distances > MAX_POSITION_DELTA] = np.inf

    reverse = np.argmin(descriptor_distances, axis=0)

    candidates: list[tuple[int, int]] = []

    for source_index, row in enumerate(descriptor_distances):
        eligible = np.flatnonzero(np.isfinite(row))

        if len(eligible) < 2:
            continue

        ranked = eligible[np.argsort(row[eligible])]

        best, runner_up = int(ranked[0]), int(ranked[1])

        if row[best] >= SPOT_MATCH_RATIO * max(float(row[runner_up]), 1e-10):
            continue

        if int(reverse[best]) != source_index:
            continue

        candidates.append((source_index, best))

    return candidates


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

    pairs = candidate_matches(source.points, target.points, aligned if alignment_applied else None)

    if alignment_applied:
        spatial_pairs, spatial_median = spatial_matches(aligned, target.points)
        spatial_source_recall = len(spatial_pairs) / max(1, len(source.points))
        spatial_target_recall = len(spatial_pairs) / max(1, len(target.points))
        spatial_repeatability = len(spatial_pairs) / max(
            1, min(len(source.points), len(target.points))
        )
    else:
        spatial_pairs = []
        spatial_median = None
        spatial_source_recall = None
        spatial_target_recall = None
        spatial_repeatability = None

    statistics: ComparisonRow = {
        "source": "",
        "target": "",
        "same_individual": False,
        "source_spots": len(source.points),
        "target_spots": len(target.points),
        "tentative_matches": len(pairs),
        "ransac_attempted": False,
        "inliers": 0,
        "inlier_ratio": None,
        "median_residual_px": None,
        "source_coverage": 0.0,
        "target_coverage": 0.0,
        "alignment_inliers": alignment_inliers,
        "alignment_applied": alignment_applied,
        "spatial_matches": len(spatial_pairs),
        "spatial_source_recall": spatial_source_recall,
        "spatial_target_recall": spatial_target_recall,
        "spatial_repeatability": spatial_repeatability,
        "spatial_median_error_px": spatial_median,
    }

    inlier_mask = [False] * len(pairs)

    if len(pairs) < MIN_RANSAC_MATCHES:
        return MatchResult(statistics, pairs, inlier_mask, spatial_pairs)

    source_points = np.asarray([source.points[i] for i, _ in pairs], dtype=np.float32)
    target_points = np.asarray([target.points[j] for _, j in pairs], dtype=np.float32)

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
        return MatchResult(statistics, pairs, inlier_mask, spatial_pairs)

    accepted = mask.ravel().astype(bool)

    residuals = np.linalg.norm(
        source_points @ matrix[:, :2].T + matrix[:, 2] - target_points,
        axis=1,
    )

    count = int(accepted.sum())

    statistics["inliers"] = count
    statistics["inlier_ratio"] = count / len(pairs)
    statistics["median_residual_px"] = float(np.median(residuals[accepted])) if count else None
    statistics["source_coverage"] = count / max(1, len(source.points))
    statistics["target_coverage"] = count / max(1, len(target.points))

    return MatchResult(statistics, pairs, accepted.tolist(), spatial_pairs)


def draw_matches(
    source: Sample, target: Sample, result: MatchResult, *, spatial: bool = False
) -> Image:
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

    if spatial:
        display_pairs = result.spatial_pairs
        accepted_flags = [True] * len(display_pairs)
    else:
        display_pairs = result.pairs
        accepted_flags = result.inlier_mask

    for (source_index, target_index), accepted in zip(display_pairs, accepted_flags, strict=True):
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

        spatial_figure = draw_matches(source, target, result, spatial=True)

        p = output / name.replace(".png", "--spatial.png")
        if not cv.imwrite(str(p), spatial_figure):
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
        "max_position_delta": MAX_POSITION_DELTA,
        "minimum_alignment_inliers": MIN_ALIGNMENT_INLIERS,
        "maximum_spatial_distance_px": MAX_SPATIAL_DISTANCE_PX,
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
