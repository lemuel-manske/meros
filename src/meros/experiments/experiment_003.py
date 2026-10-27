"""Experiment 003: visual feature matching with geometric verification."""

from __future__ import annotations

import csv

import cv2 as cv
import numpy as np

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, TypedDict, cast

from meros.domain import VideoAnnotation
from meros.experiments.common import experiment_output, Image, Points
from meros.ml.spot_sift import extract_spot_sift
from meros.project import Project


type FrameLabel = Literal["reference", "target"]
type SampleKey = tuple[str, FrameLabel]


# Sample selection
FRAME_DELTA = 30  # how many frames forward/backward to look for a comparison frame?
MIN_QUALITY = 3

# Visual feature matching
MIN_GEOMETRY_MATCHES = 4
SIFT_SPOT_PERMISSIVENESS = 0.3  # 0 = only spots; 1 = allow all extra features
SIFT_MATCH_RATIO = 0.7  # best SIFT match must be clearly better than the second-best
SIFT_RANSAC_THRESHOLD_PX = 4.0


class ComparisonRow(TypedDict):
    source: str
    target: str
    same_individual: bool
    mutual_matches: int
    inliers: int
    inlier_ratio: float | None
    identity_score: float


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
    track_id: str
    video_id: str
    annotation: VideoAnnotation

    @property
    def sample_id(self) -> str:
        return f"{self.annotation.start_frame}-{self.annotation.end_frame}"


@dataclass(frozen=True)
class MatchResult:
    statistics: ComparisonRow
    pairs: list[tuple[int, int]]
    accepted: list[bool]


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
        "mutual_matches": len(pairs),
        "inliers": 0,
        "inlier_ratio": None,
        "identity_score": 0.0,
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

    if matrix is None or mask is None:
        return MatchResult(statistics, pairs, accepted)

    accepted = mask.ravel().astype(bool).tolist()

    inliers = sum(accepted)

    statistics["inliers"] = inliers
    statistics["inlier_ratio"] = inliers / len(pairs)
    statistics["identity_score"] = (
        inliers / min(source_count, target_count) if source_count and target_count else 0.0
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


def eligible_selections(project: Project) -> list[EligibleSelection]:
    videos = {video.video_id: video for video in project.metadata.read_videos()}

    eligible: list[EligibleSelection] = []

    for vid in project.metadata.read_videos():
        for ind in vid.individuals:
            video = videos.get(vid.video_id)

            if video is None:
                raise ValueError(f"Video metadata missing for {vid.video_id}")

            annotations = video.matching_annotations(
                viewpoint="frontal",
                min_quality=MIN_QUALITY,
                track_id=ind.track_id,
            )

            for annotation in annotations:
                eligible.append(
                    EligibleSelection(
                        individual_id=ind.individual_id,
                        track_id=ind.track_id,
                        video_id=vid.video_id,
                        annotation=annotation,
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

        vid = project.metadata.read_video(selection.video_id)

        annotations = vid.matching_annotations(
            viewpoint="frontal",
            min_quality=3,
        )

        for annotation in annotations:
            reference = annotation.start_frame
            target = annotation.end_frame

            frames.append(("reference", int(reference)))
            frames.append(("target", int(target)))

            for label, frame_index in frames:
                frame = cast(
                    Image,
                    project.media.read_masked_crop(
                        selection.video_id,
                        selection.track_id,
                        frame_index,
                    ),
                )

                mask = np.asarray(
                    (frame[:, :, 3] == 255).astype(np.uint8) * 255,
                    dtype=np.uint8,
                )

                clean = cast(Image, cv.cvtColor(frame, cv.COLOR_BGRA2BGR))
                features = extract_spot_sift(frame, mask, permissiveness=SIFT_SPOT_PERMISSIVENESS)

                annotated = clean.copy()

                for x, y in features.spots:
                    cv.circle(
                        annotated,
                        (round(float(x)), round(float(y))),
                        4,
                        (0, 255, 255),
                        1,
                    )

                name = f"{selection.sample_id}--spots.png"
                p = output / name

                if not cv.imwrite(p, annotated):
                    raise OSError(f"Could not write {p}")

                samples[selection.sample_id, label] = Sample(
                    clean,
                    features.spots,
                    mask,
                    features.keypoints,
                    features.descriptors,
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

        name = f"{source_selection.sample_id}--vs--{target_selection.sample_id}.png"

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
    run(Project.open(), experiment_output("003"))


if __name__ == "__main__":
    main()
