"""Experiment 003: head-spot geometry with affine RANSAC verification."""

from __future__ import annotations

import argparse
import csv
import json
import uuid

import cv2 as cv
import numpy as np

from dataclasses import dataclass
from datetime import datetime, timezone
from numpy.typing import NDArray
from pathlib import Path
from scipy.spatial import KDTree
from typing import Literal, NotRequired, TypedDict, cast


type VideoId = Literal["h01", "h02"]
type FrameLabel = Literal["reference", "offset"]

type Image = NDArray[np.uint8]
type Points = NDArray[np.float32]

type Roi = tuple[int, int, int, int]

type SampleKey = tuple[VideoId, FrameLabel]

VIDEO_IDS: tuple[VideoId, ...] = ("h01", "h02")
NEIGHBORS = 6
MAX_SPOTS = 180
MIN_RANSAC_MATCHES = 8


class TrackData(TypedDict):
    video_id: str
    track_id: str
    start_frame: int
    end_frame: int
    reference_frame: int


class IndividualData(TypedDict):
    individual_id: str
    tracks: list[TrackData]


class VideoData(TypedDict):
    video_id: str
    path: str
    mirrored: NotRequired[bool]


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
class MatchResult:
    statistics: ComparisonRow
    pairs: list[tuple[int, int]]
    inlier_mask: list[bool]


def validate_exclusions(
    exclusions: list[Roi],
    width: int,
    height: int,
) -> None:
    for x, y, w, h in exclusions:
        if not (0 <= x < width and 0 <= y < height and w > 0 and h > 0):
            raise ValueError(f"Invalid watermark exclusion: {(x, y, w, h)}")

        if x + w > width or y + h > height:
            raise ValueError(f"Watermark exclusion outside frame: {(x, y, w, h)}")


def detect_spots(
    image: Image,
    roi: Roi | None,
    exclusions: list[Roi] | None = None,
    origin: tuple[int, int] = (0, 0),
) -> tuple[Points, Image]:
    height, width = image.shape[:2]

    x, y, w, h = roi if roi is not None else (0, 0, width, height)

    if not (0 <= x < width and 0 <= y < height and w > 0 and h > 0):
        raise ValueError(f"Invalid ROI: {roi}")

    if x + w > width or y + h > height:
        raise ValueError(f"ROI outside image: {roi}")

    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected a uint8 BGRA masked crop")

    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 masked crop")

    exclusion_boxes = exclusions or []

    # Exclusion boxes are specified in original full-frame coordinates.
    full_x, full_y = origin

    valid = image[:, :, 3] == 255

    for ex, ey, ew, eh in exclusion_boxes:
        local_x1 = max(0, ex - full_x)
        local_y1 = max(0, ey - full_y)
        local_x2 = min(width, ex + ew - full_x)
        local_y2 = min(height, ey + eh - full_y)

        if local_x1 < local_x2 and local_y1 < local_y2:
            valid[local_y1:local_y2, local_x1:local_x2] = False

    valid = cv.erode(
        valid.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)

    valid[:y, :] = False
    valid[y + h :, :] = False
    valid[:, :x] = False
    valid[:, x + w :] = False

    gray = cv.cvtColor(image[:, :, :3], cv.COLOR_BGR2GRAY)

    # A neutral fill avoids black transparent pixels polluting CLAHE and thresholding.
    active = gray[valid]

    if active.size == 0:
        return np.empty((0, 2), dtype=np.float32), cv.cvtColor(
            image, cv.COLOR_BGRA2BGR
        )

    filled = gray.copy()

    filled[~valid] = int(np.median(active))

    gray = filled

    enhanced = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    binary = cv.adaptiveThreshold(
        enhanced,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        35,
        8,
    )

    count, _, stats, centers = cv.connectedComponentsWithStats(binary, connectivity=8)

    found: list[tuple[float, float, int]] = []

    for index in range(1, count):
        left, top, component_width, component_height, area = map(int, stats[index])

        if not (5 <= area <= max(10, int(w * h * 0.001))):
            continue

        if component_width < 2 or component_height < 2:
            continue

        if left == 0 or top == 0 or left + component_width >= w or top + component_height >= h:
            continue

        aspect = component_width / component_height
        density = area / (component_width * component_height)

        if not (0.35 <= aspect <= 2.8 and 0.25 <= density <= 0.94):
            continue

        if not np.all(valid[top : top + component_height, left : left + component_width]):
            continue

        center_x, center_y = centers[index]

        found.append((float(center_x + x), float(center_y + y), area))

    found.sort(key=lambda item: (-item[2], item[1], item[0]))

    points = np.asarray(
        [(center_x, center_y) for center_x, center_y, _ in found[:MAX_SPOTS]],
        dtype=np.float32,
    ).reshape(-1, 2)

    annotated = cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    cv.rectangle(annotated, (x, y), (x + w - 1, y + h - 1), (255, 180, 0), 2)

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


def read_metadata(
    root: Path,
) -> tuple[dict[str, VideoData], dict[VideoId, tuple[str, TrackData]]]:
    raw_videos = json.loads((root / "metadata/videos/videos.json").read_text())
    raw_individuals = json.loads((root / "metadata/individuals/individuals.json").read_text())

    videos = {video["video_id"]: video for video in cast(list[VideoData], raw_videos)}

    selections: dict[VideoId, tuple[str, TrackData]] = {}

    for individual in cast(list[IndividualData], raw_individuals):
        for track in individual["tracks"]:
            if track["video_id"] not in VIDEO_IDS:
                continue

            video_id = cast(VideoId, track["video_id"])

            selections[video_id] = (individual["individual_id"], track)

    if set(selections) != set(VIDEO_IDS):
        raise ValueError("Need both h01 and h02 in individuals.json")

    if selections["h01"][0] == selections["h02"][0]:
        raise ValueError("h01 and h02 must be different individuals")

    return videos, selections


def offset_frame(track: TrackData, delta: int) -> int:
    reference = track["reference_frame"]

    alternative = reference + delta

    if alternative > track["end_frame"]:
        alternative = reference - delta

    if alternative < track["start_frame"]:
        raise ValueError(f"No valid offset frame for {track['video_id']}")

    return alternative


def save_image(path: Path, image: Image) -> None:
    if not cv.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def run(
    data: Path,
    output: Path,
    delta: int,
    rois: dict[VideoId, Roi | None],
    exclusions: dict[VideoId, list[Roi]],
) -> list[ComparisonRow]:
    videos, selections = read_metadata(data)

    from meros.project import Project

    project = Project.open(data)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")

    output.mkdir(parents=True)

    samples: dict[SampleKey, Sample] = {}

    for video_id in VIDEO_IDS:
        track = selections[video_id][1]

        frames: tuple[tuple[FrameLabel, int], ...] = (
            ("reference", track["reference_frame"]),
            ("offset", offset_frame(track, delta)),
        )

        for label, frame_index in frames:
            frame = cast(
                Image,
                project.media.read_masked_crop(
                    video_id,
                    track["track_id"],
                    frame_index,
                ),
            )

            # Crops are bounding-box-relative; recover their position from track metadata.
            track_metadata = project.metadata.read_track(video_id)
            observation = track_metadata.tracks[track["track_id"]].frames[str(frame_index)]
            x1, y1, _, _ = observation.bbox

            points, annotated = detect_spots(
                frame,
                rois[video_id],
                exclusions[video_id],
                origin=(x1, y1),
            )

            save_image(output / f"{video_id}_{label}_spots.png", annotated)

            samples[video_id, label] = Sample(annotated, points)

    comparisons: tuple[tuple[SampleKey, SampleKey, bool], ...] = (
        (("h01", "reference"), ("h01", "offset"), True),
        (("h02", "reference"), ("h02", "offset"), True),
        (("h01", "reference"), ("h02", "reference"), False),
    )

    rows: list[ComparisonRow] = []

    for source_key, target_key, same_individual in comparisons:
        source = samples[source_key]
        target = samples[target_key]

        result = compare(source.points, target.points)

        result.statistics["source"] = "/".join(source_key)
        result.statistics["target"] = "/".join(target_key)
        result.statistics["same_individual"] = same_individual

        figure = draw_matches(source, target, result)

        name = f"{source_key[0]}_{source_key[1]}_vs_{target_key[0]}_{target_key[1]}.png"

        save_image(output / name, figure)

        rows.append(result.statistics)

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ComparisonRow.__annotations__))

        writer.writeheader()

        writer.writerows(rows)

    provenance: dict[str, object] = {
        "delta": delta,
        "rois": rois,
        "exclusions": exclusions,
        "video_metadata": {
            video_id: {"mirrored": videos[video_id].get("mirrored", False)}
            for video_id in VIDEO_IDS
        },
    }

    (output / "run.json").write_text(json.dumps(provenance, indent=2))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--roi-h01", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--roi-h02", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--exclude-h01", nargs=4, type=int, action="append",
                        metavar=("X", "Y", "W", "H"), default=[])
    parser.add_argument("--exclude-h02", nargs=4, type=int, action="append",
                        metavar=("X", "Y", "W", "H"), default=[])
    parser.add_argument("--delta", type=int, default=30)

    args = parser.parse_args()

    if args.delta <= 0:
        parser.error("--delta must be positive")

    rois: dict[VideoId, Roi | None] = {
        "h01": cast(Roi, tuple(args.roi_h01)) if args.roi_h01 is not None else None,
        "h02": cast(Roi, tuple(args.roi_h02)) if args.roi_h02 is not None else None,
    }

    output = args.output or Path("results/003") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )

    exclusions: dict[VideoId, list[Roi]] = {
        "h01": [cast(Roi, tuple(box)) for box in args.exclude_h01],
        "h02": [cast(Roi, tuple(box)) for box in args.exclude_h02],
    }

    rows = run(args.data, output, args.delta, rois, exclusions)

    print(f"Exp-003: wrote {len(rows)} comparisons to {output}")


if __name__ == "__main__":
    main()
