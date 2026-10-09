"""Completion contracts tested against real images and serialized metadata."""

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import cv2 as cv
import pytest

from meros.adapters.fs_storage import write_json_atomic
from meros.domain import Individual, TrackSelection
from meros.processing.align_masked_crops import alignments_complete
from meros.processing.build_composites import composites_complete
from meros.processing.enhance_composites import enhanced_composites_complete
from meros.processing.extract_frames import frames_complete
from meros.processing.extract_masked_crops import masked_crops_complete
from meros.processing.extract_tracks import tracks_complete
from meros.project import Project
from tests.support import ProjectFixture


CHECKS: dict[str, Callable[[Project], bool]] = {
    "frames": frames_complete,
    "tracks": tracks_complete,
    "masked_crops": masked_crops_complete,
    "alignments": alignments_complete,
    "composites": composites_complete,
    "enhanced_composites": enhanced_composites_complete,
}


def artifact_path(fixture: ProjectFixture, stage: str) -> Path:
    project = fixture.project

    track = fixture.selections()[0]

    if stage == "frames":
        return project.media.frame_path("a", 0)

    if stage == "tracks":
        return project.metadata.paths.track("a")

    if stage == "masked_crops":
        return project.media.paths.masked_crop("a", "0", 0)

    if stage == "alignments":
        return project.metadata.paths.alignment(*track.id)

    if stage == "composites":
        return project.media.paths.composite(track)

    return project.media.paths.composite_enhanced(track)


@pytest.mark.parametrize("stage", CHECKS)
def test_complete_outputs_are_reusable(prepared_project: ProjectFixture, stage: str):
    assert CHECKS[stage](prepared_project.project)


@pytest.mark.parametrize("stage", CHECKS)
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_outputs_are_incomplete(
    prepared_project: ProjectFixture, stage: str, damage: str
):
    path = artifact_path(prepared_project, stage)

    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"not an image or JSON")

    assert not CHECKS[stage](prepared_project.project)


@pytest.mark.parametrize("stage", CHECKS)
def test_empty_inputs_are_incomplete(prepared_project: ProjectFixture, stage: str):
    if stage in {"frames", "tracks", "masked_crops"}:
        write_json_atomic(prepared_project.project.metadata.paths.videos(), [])
    else:
        prepared_project.empty_selections()

    assert not CHECKS[stage](prepared_project.project)


@pytest.mark.parametrize(
    "stage,damage",
    [
        (stage, damage)
        for stage in ["frames", "masked_crops", "composites", "enhanced_composites"]
        for damage in ["dimensions", "channels", "transparent"]
        if stage != "frames" or damage != "transparent"
    ],
)
def test_images_must_match_output_contract(
    prepared_project: ProjectFixture, stage: str, damage: str
):
    path = artifact_path(prepared_project, stage)

    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)

    assert image is not None

    if damage == "dimensions":
        image = image[:1]
    elif damage == "channels":
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        image[:, :, 3] = 0

    assert cv.imwrite(str(path), image)

    assert not CHECKS[stage](prepared_project.project)


@pytest.mark.parametrize(
    "damage",
    [
        "count",
        "video",
        "seed",
        "initial_frame",
        "empty",
        "outside",
        "negative_area",
        "bbox",
    ],
)
def test_track_metadata_must_describe_processed_sequence(
    prepared_project: ProjectFixture, damage: str
):
    path = artifact_path(prepared_project, "tracks")

    payload = json.loads(path.read_text())

    track = payload["tracks"]["0"]

    if damage == "count":
        payload["processed_frame_count"] = 2
    elif damage == "video":
        payload["video_id"] = "other"
    elif damage == "seed":
        track["initial_bbox"][0] += 1
    elif damage == "initial_frame":
        track["initial_frame"] = 1
    elif damage == "empty":
        track["frames"] = {}
    elif damage == "outside":
        track["frames"]["9"] = track["frames"]["0"]
    elif damage == "negative_area":
        track["frames"]["0"]["mask_area"] = -1
    else:
        track["frames"]["0"]["bbox"] = [10, 10, 0, 0]

    write_json_atomic(path, payload)

    assert not tracks_complete(prepared_project.project)


@pytest.mark.parametrize(
    "damage",
    [
        "interval",
        "reference",
        "video",
        "duplicate",
        "missing",
        "status",
        "matrix_shape",
        "matrix_nan",
        "matrix_singular",
        "inliers",
    ],
)
def test_alignment_metadata_must_describe_selection(prepared_project: ProjectFixture, damage: str):
    path = artifact_path(prepared_project, "alignments")

    payload = json.loads(path.read_text())

    row = payload["frames"][0]

    if damage == "interval":
        payload["end_frame"] += 1
    elif damage == "reference":
        payload["reference_frame"] = 1
    elif damage == "video":
        payload["video_id"] = "other"
    elif damage == "duplicate":
        payload["frames"].append(row.copy())
    elif damage == "missing":
        payload["frames"].pop()
    elif damage == "status":
        row["status"] = "unknown"
    elif damage == "matrix_shape":
        row["source_to_reference"] = [[1, 2]]
    elif damage == "matrix_nan":
        row["source_to_reference"][0][0] = float("nan")
    elif damage == "matrix_singular":
        row["source_to_reference"] = [[0, 0, 0], [0, 0, 0]]
    else:
        row["inliers"] = row["matches"] + 1

    write_json_atomic(path, payload)

    assert not alignments_complete(prepared_project.project)


def test_rejected_alignment_is_a_complete_result(prepared_project: ProjectFixture):
    path = artifact_path(prepared_project, "alignments")

    payload = json.loads(path.read_text())

    payload["frames"][0].update(status="insufficient_matches", source_to_reference=None, inliers=0)

    write_json_atomic(path, payload)

    assert alignments_complete(prepared_project.project)


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_count", "three"),
        ("frame_count", 0),
        ("mirrored", "false"),
    ],
)
def test_malformed_video_metadata_is_incomplete(
    prepared_project: ProjectFixture, field: str, value: object
):
    path = prepared_project.project.metadata.paths.videos()

    payload = json.loads(path.read_text())

    payload[0][field] = value

    write_json_atomic(path, payload)

    assert not frames_complete(prepared_project.project)


@pytest.mark.parametrize(
    "damage",
    ["area_overflow", "frame_type", "frames_type", "tracks_type", "count_type", "outside_bbox"],
)
def test_malformed_track_fields_are_incomplete(prepared_project: ProjectFixture, damage: str):
    path = artifact_path(prepared_project, "tracks")

    payload = json.loads(path.read_text())

    track = payload["tracks"]["0"]

    if damage == "area_overflow":
        track["frames"]["0"]["mask_area"] = 999999
    elif damage == "frame_type":
        track["initial_frame"] = "zero"
    elif damage == "frames_type":
        track["frames"] = None
    elif damage == "tracks_type":
        payload["tracks"] = []
    elif damage == "count_type":
        payload["processed_frame_count"] = "three"
    else:
        track["frames"]["0"]["bbox"] = [0, 0, 999, 999]

    write_json_atomic(path, payload)

    assert not tracks_complete(prepared_project.project)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_to_reference", None),
        ("source_to_reference", [[-1, 0, 0], [0, 1, 0]]),
        ("inlier_ratio", 0.1),
        ("inlier_hull_fraction", 0.01),
        ("inlier_ratio", float("nan")),
        ("matches", "many"),
        ("frame_idx", []),
    ],
)
def test_malformed_or_unqualified_alignment_is_incomplete(
    prepared_project: ProjectFixture, field: str, value: object
):
    path = artifact_path(prepared_project, "alignments")

    payload = json.loads(path.read_text())

    payload["frames"][0][field] = value

    write_json_atomic(path, payload)

    assert not alignments_complete(prepared_project.project)


def test_frame_cache_rejects_extra_and_non_numeric_files(prepared_project: ProjectFixture):
    project = prepared_project.project

    path = project.media.frames_path("a") / "preview.jpg"

    path.write_bytes(project.media.frame_path("a", 0).read_bytes())

    assert not frames_complete(project)

    path.rename(path.with_name("9.jpg"))

    assert not frames_complete(project)


def test_single_frame_selection_needs_no_alignment_rows(prepared_project: ProjectFixture):
    project = prepared_project.project

    selection = TrackSelection("a", "0", 0, 0, 0)

    assert project.metadata.individuals_path is not None

    write_json_atomic(
        project.metadata.individuals_path, [asdict(Individual("single", [selection]))]
    )

    project.metadata.write_alignment(selection.id, [])

    assert alignments_complete(project)


@pytest.mark.parametrize("payload", [[], {"a": []}, {"a": {"0": [1, 2]}}])
def test_malformed_seed_metadata_is_incomplete(prepared_project: ProjectFixture, payload: object):
    write_json_atomic(prepared_project.project.metadata.paths.bboxes(), payload)

    assert not tracks_complete(prepared_project.project)

    assert not masked_crops_complete(prepared_project.project)


@pytest.mark.parametrize("stage", ["masked_crops", "composites", "enhanced_composites"])
def test_generated_images_require_binary_alpha(prepared_project: ProjectFixture, stage: str):
    path = artifact_path(prepared_project, stage)

    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)

    assert image is not None

    image[0, 0, 3] = 127

    assert cv.imwrite(str(path), image)

    assert not CHECKS[stage](prepared_project.project)


def test_crop_coverage_must_match_observation(prepared_project: ProjectFixture):
    path = artifact_path(prepared_project, "masked_crops")

    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)

    assert image is not None

    image[0, 0] = 0

    assert cv.imwrite(str(path), image)

    assert not masked_crops_complete(prepared_project.project)
