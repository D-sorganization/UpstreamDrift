from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pytest
from tests.unit.motion_pipeline.preprocessing._local_fixtures import make_keypoint_sequence
from src.shared.python.motion_pipeline.preprocessing.apply import apply_preprocessing
from src.shared.python.motion_pipeline.preprocessing.gap_fill import GapFillStrategy
from src.shared.python.motion_pipeline.preprocessing.filter import FilterType
from src.shared.python.motion_pipeline.preprocessing.normalize import UpAxis


@dataclass
class DummyStepConfig:
    name: str
    params: dict[str, Any] | None = None


def test_apply_preprocessing_gap_fill() -> None:
    # Butterworth padding requires more frames, but gap fill is fine with 10.
    seq = make_keypoint_sequence(num_frames=10)
    # Test valid strategy
    step_linear = DummyStepConfig(name="GapFill", params={"strategy": "linear", "max_gap": 5})
    res_linear = apply_preprocessing(seq, [step_linear])
    assert res_linear is not None

    # Test invalid strategy (suppressed ValueError, falls back to None -> LINEAR in post_init)
    step_invalid = DummyStepConfig(name="Gap-Fill", params={"strategy": "invalid"})
    res_invalid = apply_preprocessing(seq, [step_invalid])
    assert res_invalid is not None


def test_apply_preprocessing_filter() -> None:
    # Padlen requirement is 12 (order 3, ntaps=4, ntaps*3=12), so we need 30 frames
    seq = make_keypoint_sequence(num_frames=30)
    step_filter = DummyStepConfig(
        name="filter_step",
        params={"filter_type": "butterworth", "cutoff": 5.0, "order": 3, "fps": 60.0},
    )
    res = apply_preprocessing(seq, [step_filter])
    assert res is not None

    # Test invalid filter type (falls back to BUTTERWORTH)
    step_invalid = DummyStepConfig(name="filter", params={"filter_type": "invalid"})
    res_invalid = apply_preprocessing(seq, [step_invalid])
    assert res_invalid is not None


def test_apply_preprocessing_resample() -> None:
    seq = make_keypoint_sequence(num_frames=10)
    step_resample = DummyStepConfig(
        name="ResampleStep", params={"target_fps": 30.0, "source_fps": 60.0}
    )
    res = apply_preprocessing(seq, [step_resample])
    assert res is not None


def test_apply_preprocessing_normalize() -> None:
    seq = make_keypoint_sequence(num_frames=10)
    step_norm = DummyStepConfig(
        name="Normalize-Step",
        params={
            "target_up": UpAxis.Z_UP,
            "source_up": UpAxis.Y_UP,
            "center_origin": True,
        },
    )
    res = apply_preprocessing(seq, [step_norm])
    assert res is not None


def test_apply_preprocessing_unknown_step() -> None:
    seq = make_keypoint_sequence(num_frames=10)
    step_unknown = DummyStepConfig(name="UnknownStep")
    with pytest.raises(ValueError, match="Unknown preprocessing step name: UnknownStep"):
        apply_preprocessing(seq, [step_unknown])
