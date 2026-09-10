"""Analytical calibration and recording contracts for model-independent inputs."""

import numpy as np
import pytest

from src.shared.python.biomechanics.model_bindings import (
    ModelBinding,
    SegmentBinding,
    TrajectoryRecorder,
    model_binding_from_dict,
)

pytestmark = pytest.mark.unit


def test_calibration_transforms_origin_and_preserves_missing_frames():
    calibration = np.eye(4)
    calibration[0, 3] = 0.25
    binding = ModelBinding(
        source="synthetic",
        world_frame="right-handed Z-up",
        length_scale=0.01,
        segments={"pelvis": SegmentBinding("hip", calibration=calibration)},
    )
    recorder = TrajectoryRecorder(binding)
    transform = np.eye(4)
    transform[0, 3] = 100
    recorder.append(0, {"hip": transform})
    recorder.append(1, {})
    payload = recorder.to_payload()
    assert payload["segments"]["pelvis"]["positions"][0] == [1.25, 0, 0]
    assert payload["segments"]["pelvis"]["positions"][1] == [None] * 3
    assert payload["source"] == "synthetic"


def test_snapshot_is_owned_and_time_must_increase():
    recorder = TrajectoryRecorder(
        ModelBinding("test", "world", {"a": SegmentBinding("a")})
    )
    transform = np.eye(4)
    recorder.append(0, {"a": transform})
    transform[0, 3] = 7
    recorder.append(1, {"a": transform})
    assert recorder.to_payload()["segments"]["a"]["positions"][0][0] == 0
    with pytest.raises(ValueError, match="increasing"):
        recorder.append(1, {"a": transform})


@pytest.mark.parametrize("scale", [0, -1, float("nan")])
def test_invalid_units_rejected(scale):
    with pytest.raises(ValueError, match="length_scale"):
        ModelBinding("test", "world", {"a": SegmentBinding("a")}, length_scale=scale)


def test_reflection_and_nonrigid_transform_rejected():
    calibration = np.eye(4)
    calibration[0, 0] = -1
    with pytest.raises(ValueError, match="proper rotation"):
        SegmentBinding("a", calibration=calibration)


def test_failed_frame_is_atomic():
    recorder = TrajectoryRecorder(
        ModelBinding("test", "world", {"a": SegmentBinding("a")})
    )
    with pytest.raises(ValueError):
        recorder.append(0, {"a": np.zeros((4, 4))})
    recorder.append(0, {"a": np.eye(4)})
    recorder.append(1, {"a": np.eye(4)})
    assert recorder.to_payload()["times"] == [0, 1]


def test_binding_json_parser_is_strict():
    binding = model_binding_from_dict(
        {"source": "test", "world_frame": "Z-up", "segments": {"a": {"link": "link_a"}}}
    )
    assert binding.segments["a"].link == "link_a"
    with pytest.raises(ValueError, match="Unknown"):
        model_binding_from_dict(
            {"source": "test", "world_frame": "Z-up", "segments": {}, "typo": 2}
        )
