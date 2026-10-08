"""Compare algorithms with the immutable pre-refactor implementation."""

import subprocess
import sys
import types
from dataclasses import asdict

import cv2 as cv
import numpy as np
import pytest

from meros.processing import align_masked_crops, build_composites, enhance_composites
from meros.processing.match_images import match_images

BASELINE = "386bdcec3b00ea348c523f1d3904c41210f175ba"


def legacy(name: str) -> types.ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:src/meros/cmd/{name}.py"], text=True
    )

    source = (
        source.replace("src.meros", "meros")
        .replace("IndividualTrack", "TrackSelection")
        .replace("IndividualId", "TrackSelectionKey")
        .replace("from meros import media, metadata", "")
    )

    module = types.ModuleType(f"_baseline_{name}")

    sys.modules[module.__name__] = module

    exec(compile(source, f"<baseline:{name}>", "exec"), module.__dict__)

    return module


@pytest.fixture
def textured_crop():
    random = np.random.default_rng(42)

    image = random.integers(0, 256, (160, 200, 4), dtype=np.uint8)

    image[:, :, 3] = 255

    image[:8] = 0

    return image


def test_matching_statistics_preserved(textured_crop: np.ndarray) -> None:
    original = legacy("match_composites")

    target = cv.warpAffine(textured_crop, np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]]), (200, 160))

    for a, b in [(textured_crop, textured_crop), (textured_crop, target)]:
        _, expected = original.match_composites(a, b)

        _, actual = match_images(a, b)

        result = asdict(actual)

        result.pop("ransac_attempted")

        assert result == asdict(expected)

        assert actual.inliers > 0


def test_enhancement_pixels_preserved(textured_crop: np.ndarray) -> None:
    expected = legacy("enhance_composites").enhance_contrast(textured_crop)

    assert np.array_equal(expected, enhance_composites.enhance_contrast(textured_crop))


def test_composite_pixels_preserved(textured_crop: np.ndarray) -> None:
    second = textured_crop.copy()

    second[20:40] = 0

    expected = legacy("build_composites").median_composite([textured_crop, second])

    assert np.array_equal(expected, build_composites.median_composite([textured_crop, second]))


def test_alignment_transforms_and_statistics_preserved(textured_crop: np.ndarray) -> None:
    expected_matrix, expected = legacy("align_masked_crops").estimate_alignment(
        textured_crop, textured_crop
    )

    actual_matrix, actual = align_masked_crops.estimate_alignment(textured_crop, textured_crop)

    assert actual_matrix is not None

    assert np.array_equal(expected_matrix, actual_matrix)

    before = asdict(expected)

    before["status"] = "accepted" if before["status"] == "candidate" else before["status"]

    assert asdict(actual) == before
