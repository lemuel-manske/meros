"""Experiment 003: visual feature matching with geometric verification."""

from __future__ import annotations

import csv

import cv2 as cv
import numpy as np

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, TypedDict, cast

from meros.domain import TrackSelection, VideoAnnotation
from meros.experiments.common import debug_img, experiment_output
from meros.project import Project
from meros.types import Image, Points


type FrameLabel = Literal["reference", "target"]

type SampleKey = tuple[str, FrameLabel]

# hard-set pairs for comparison,
# for example, to test very distant frames
pairs = {
    "h01": ("170", "510"),
    "h02": ("1000", "1140"),
    "h03": ("3290", "3416"),
    "h04": ("2817", "2937"),
}

# Sample selection
FRAME_DELTA = 30  # how many frames forward/backward to look for a comparison frame?
MIN_QUALITY = 3

# Spot detection
MAX_SPOTS = 320  # how many spots to consider per image?
SPOT_THRESHOLD_BLOCK_SIZE = 35
SPOT_THRESHOLD_C = 8

# Visual feature matching
MIN_GEOMETRY_MATCHES = 4
SPOT_REGION_RADIUS_PX = 16
SIFT_FEATURES = 2500  # how many SIFT features to extract per image?
SIFT_MATCH_RATIO = 0.7  # best SIFT match must be clearly better than the second-best
SIFT_RANSAC_THRESHOLD_PX = 4.0


class ComparisonRow(TypedDict):
    source: str
    target: str
    same_individual: bool
    source_features: int
    target_features: int
    forward_matches: int
    reverse_matches: int
    mutual_matches: int
    ransac_attempted: bool
    inliers: int
    inlier_ratio: float | None
    feature_coverage: float | None
    identity_score: float
    median_residual_px: float | None


@dataclass(frozen=True)
class Sample:
    image: Image
    points: Points
    mask: Image
    keypoints: tuple[cv.KeyPoint, ...]
    descriptors: Points | None


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
    accepted: list[bool]


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

    gray = cv.cvtColor(image[:, :, :3], cv.COLOR_BGR2GRAY)  # gray scale
    gray_active = gray[valid]  # only consider pixels inside the valid mask

    debug_img(gray)

    # draw the annotated image in BGR, ignoring the alpha channel
    # for demonstration purposes
    annotated = cast(Image, cv.cvtColor(image, cv.COLOR_BGRA2BGR))

    if gray_active.size == 0:  # no valid pixels to detect spots
        return np.empty((0, 2), dtype=np.float32), annotated

    filled = gray.copy()
    filled[~valid] = int(
        np.median(gray_active)
    )  # fill invalid pixels with the median of valid pixels

    debug_img(filled)

    # enhance the gray image contrast
    enhanced = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filled)

    debug_img(enhanced)

    # adaptative limiar so that spots are detected even in low-contrast regions
    binary = cv.adaptiveThreshold(
        enhanced,
        255,  # maxValue
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        SPOT_THRESHOLD_BLOCK_SIZE,  # blockSize
        SPOT_THRESHOLD_C,  # C
    )

    # filter out invalid; never let invalid or transparent pixels become candidate spots.
    binary[~valid] = 0

    debug_img(binary)

    count, _, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)

    found: list[tuple[float, float, int]] = []

    for index in range(1, count):  # skip the background
        left, top, component_width, component_height, area = map(int, stats[index])

        # remove tiny spots and spots that are too large relative to the image size
        if not (5 <= area <= max(10, int(width * height * 0.001))):
            continue

        # remove spots that are too close to the image edge, which are likely artifacts
        if component_width < 2 or component_height < 2:
            continue

        # remove spots that are too close to the image edge, which are likely artifacts
        if (
            left == 0
            or top == 0
            or left + component_width >= width
            or top + component_height >= height
        ):
            continue

        aspect = component_width / component_height
        density = area / (component_width * component_height)

        # heuristics: should not be too elongated or sparse
        if not (0.35 <= aspect <= 2.8 and 0.25 <= density <= 0.94):
            continue

        # filter out spots that are not fully valid; this avoids spots that are partially occluded or on the edge of the mask
        if not np.all(valid[top : top + component_height, left : left + component_width]):
            continue

        center_x, center_y = centroids[index]

        found.append((float(center_x), float(center_y), area))

    # sort out and get MAX_SPOTS best spots
    found.sort(key=lambda item: (-item[2], item[1], item[0]))
    points = np.asarray(
        [(center_x, center_y) for center_x, center_y, _ in found[:MAX_SPOTS]],
        dtype=np.float32,
    ).reshape(-1, 2)

    for center_x, center_y in points:
        center = (round(float(center_x)), round(float(center_y)))

        cv.circle(annotated, center, 4, (0, 255, 255), 1)

    return points, annotated


def extract_features(
    image: Image, mask: Image, points: Points
) -> tuple[tuple[cv.KeyPoint, ...], Points | None]:
    region_mask = np.zeros_like(mask)

    for x, y in points:
        cv.circle(
            region_mask,
            (round(float(x)), round(float(y))),
            SPOT_REGION_RADIUS_PX,
            255,
            -1,
        )

    region_mask = cv.bitwise_and(region_mask, mask)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT.create(nfeatures=SIFT_FEATURES)
    keypoints, descriptions = sift.detectAndCompute(gray, region_mask)

    debug_img(
        cv.drawKeypoints(
            image,
            keypoints,
            outImage=None,  # pyright: ignore
            color=None,  # pyright: ignore
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )  # pyright: ignore
    )

    return tuple(keypoints), (
        np.asarray(descriptions, dtype=np.float32) if descriptions is not None else None
    )


def ratio_matches(source: Points, target: Points) -> dict[int, int]:
    if len(source) < 2 or len(target) < 2:
        return {}

    matcher = cv.BFMatcher(cv.NORM_L2)

    return {
        i: int(pairs[0].trainIdx)
        for i, pairs in enumerate(matcher.knnMatch(source, target, k=2))
        if len(pairs) == 2 and pairs[0].distance < SIFT_MATCH_RATIO * pairs[1].distance
    }


def compare(source: Sample, target: Sample) -> MatchResult:
    source_desc = source.descriptors
    target_desc = target.descriptors

    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}

    if source_desc is not None and target_desc is not None:
        forward = ratio_matches(source_desc, target_desc)
        reverse = ratio_matches(target_desc, source_desc)

    pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]

    source_count = len(source.keypoints)
    target_count = len(target.keypoints)

    statistics: ComparisonRow = {
        "source": "",
        "target": "",
        "same_individual": False,
        "source_features": source_count,
        "target_features": target_count,
        "forward_matches": len(forward),
        "reverse_matches": len(reverse),
        "mutual_matches": len(pairs),
        "ransac_attempted": False,
        "inliers": 0,
        "inlier_ratio": None,
        "feature_coverage": None,
        "identity_score": 0.0,
        "median_residual_px": None,
    }

    accepted = [False] * len(pairs)

    if len(pairs) < MIN_GEOMETRY_MATCHES:
        return MatchResult(statistics, pairs, accepted)

    source_points = np.asarray([source.keypoints[i].pt for i, _ in pairs], dtype=np.float32)
    target_points = np.asarray([target.keypoints[j].pt for _, j in pairs], dtype=np.float32)

    cv.setRNGSeed(0)

    matrix, mask = cv.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv.RANSAC,
        ransacReprojThreshold=SIFT_RANSAC_THRESHOLD_PX,
        maxIters=3000,
        confidence=0.99,
    )

    statistics["ransac_attempted"] = True

    if matrix is None or mask is None:
        return MatchResult(statistics, pairs, accepted)

    accepted = mask.ravel().astype(bool).tolist()
    inliers = sum(accepted)

    residuals = np.linalg.norm(
        source_points @ matrix[:, :2].T + matrix[:, 2] - target_points,
        axis=1,
    )

    statistics["inliers"] = inliers
    statistics["inlier_ratio"] = inliers / len(pairs)
    statistics["feature_coverage"] = (
        inliers / min(source_count, target_count) if source_count and target_count else None
    )
    statistics["identity_score"] = (
        inliers / min(source_count, target_count) if source_count and target_count else 0.0
    )
    statistics["median_residual_px"] = (
        float(np.median(residuals[np.asarray(accepted, dtype=bool)])) if inliers else None
    )

    return MatchResult(statistics, pairs, accepted)


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

    for (source_index, target_index), accepted in zip(result.pairs, result.accepted, strict=True):
        source_point = (
            round(float(source.keypoints[source_index].pt[0])),
            round(float(source.keypoints[source_index].pt[1])),
        )
        target_point = (
            round(float(target.keypoints[target_index].pt[0])) + source_image.shape[1],
            round(float(target.keypoints[target_index].pt[1])),
        )

        cv.line(
            output,
            source_point,
            target_point,
            (0, 220, 0) if accepted else (40, 80, 220),
            1,
        )

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

        reference, target = pairs[selection.track.video_id]

        frames.append(("reference", int(reference)))
        frames.append(("target", int(target)))

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

            keypoints, descriptions = extract_features(clean, mask, points)

            samples[selection.sample_id, label] = Sample(
                clean, points, mask, keypoints, descriptions
            )

    comparisons: list[tuple[EligibleSelection, FrameLabel, EligibleSelection, FrameLabel]] = []

    for selection in selections:
        if (selection.sample_id, "target") in samples:
            comparisons.append((selection, "reference", selection, "target"))

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

    print(f"Wrote {len(rows)} comparisons to {output}")

    return rows


def main() -> None:
    project = Project.open()

    output = experiment_output("003")

    run(project, output)


if __name__ == "__main__":
    main()
