"""Exercise the SAM2 adapter's public API without importing GPU libraries."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from meros.domain.prediction import BinaryMask, SeedBox
from meros.external.sam2 import MaskLogits, SAM2Predictor


class ArrayTensor:
    """A tensor-compatible array used at the external library boundary."""

    def __init__(self, array: NDArray[np.float32]) -> None:
        self.array = array

    def gt(self, threshold: float) -> "ArrayTensor":
        return ArrayTensor((self.array > threshold).astype(np.float32))

    def cpu(self) -> "ArrayTensor":
        return self

    def numpy(self) -> BinaryMask:
        return self.array.astype(np.bool_)


class RecordedSam2Backend:
    def __init__(self, logits: NDArray[np.float32], object_ids: list[int]) -> None:
        self.logits = logits

        self.object_ids = object_ids

        self.state = object()

        self.loaded_path: str | None = None

        self.offload: tuple[bool, bool] | None = None

        self.seeds: list[tuple[int, int, SeedBox]] = []

    def init_state(
        self, video_path: str, *, offload_video_to_cpu: bool, offload_state_to_cpu: bool
    ) -> object:
        self.loaded_path = video_path

        self.offload = (offload_video_to_cpu, offload_state_to_cpu)

        return self.state

    def add_new_points_or_box(
        self, *, inference_state: object, frame_idx: int, obj_id: int, box: SeedBox
    ) -> object:
        assert inference_state is self.state

        self.seeds.append((frame_idx, obj_id, box))

        return None

    def propagate_in_video(
        self, state: object
    ) -> Iterator[tuple[int, list[int], list[MaskLogits]]]:
        assert state is self.state

        yield 0, self.object_ids, [ArrayTensor(self.logits)]


def test_adapter_converts_logits_and_forwards_seed_api(tmp_path: Path):
    logits = np.array([[[-1, 0, 1]]], dtype=np.float32)

    backend = RecordedSam2Backend(logits, [2])

    predictor = SAM2Predictor(backend)

    state = predictor.init_state(str(tmp_path))

    box = np.array([0, 0, 2, 1], dtype=np.float32)

    predictor.add_new_points_or_box(state, 0, 2, box)

    predictions = list(predictor.propagate_in_video(state))

    assert backend.loaded_path == str(tmp_path)

    assert backend.offload == (True, False)

    assert backend.seeds[0][:2] == (0, 2)

    assert np.array_equal(backend.seeds[0][2], box)

    assert predictions[0].frame_idx == 0

    assert np.array_equal(predictions[0].masks["2"], np.array([[False, False, True]]))

    assert predictions[0].masks["2"].shape == (1, 3)


@pytest.mark.parametrize("shape,object_ids", [((2, 2, 2), [0]), ((1, 2, 2), [0, 1])])
def test_adapter_rejects_invalid_output(shape: tuple[int, int, int], object_ids: list[int]):
    backend = RecordedSam2Backend(np.ones(shape, dtype=np.float32), object_ids)

    with pytest.raises(ValueError):
        list(SAM2Predictor(backend).propagate_in_video(backend.state))
