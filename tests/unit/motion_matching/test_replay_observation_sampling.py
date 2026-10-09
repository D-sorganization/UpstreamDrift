"""F09a boundaries between native output clocks and measured clocks."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python import motion_matching as motion_matching_api
from src.shared.python.motion_matching import (
    PositionInterpolation,
    align_native_positions_to_observations,
)

pytestmark = pytest.mark.unit


def _inputs() -> dict[str, object]:
    native_time = np.array([0.0, 0.4, 1.0])
    observation_time = np.array([0.1, 0.5, 0.9])
    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = native_time
    positions[:, 1, 1] = 2.0 * native_time
    target = np.zeros((3, 2, 3), dtype=np.float64)
    target[:, 0, 0] = observation_time + 1.0
    target[:, 1, 1] = 2.0 * observation_time + 1.0
    valid = np.ones((3, 2), dtype=bool)
    return {
        "native_output_time_s": native_time,
        "native_output_positions_m": positions,
        "observation_time_s": observation_time,
        "observation_positions_m": target,
        "observation_valid": valid,
        "native_marker_labels": ("WaistLeft", "WaistRight"),
        "observation_marker_labels": ("WaistLeft", "WaistRight"),
        "interpolation": PositionInterpolation.LINEAR_POSITION,
        "source_identity_sha256": "a" * 64,
        "native_frame_id": "world-z-up-v1",
        "observation_frame_id": "world-z-up-v1",
        "native_timebase_id": "simulation-relative-v1",
        "observation_timebase_id": "simulation-relative-v1",
    }


def test_aligns_positions_to_distinct_observation_clock_without_losing_it() -> None:
    result = align_native_positions_to_observations(**_inputs())

    np.testing.assert_array_equal(result.observation_time_s, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(result.predicted_positions_m[:, 0, 0], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(result.predicted_positions_m[:, 1, 1], [0.2, 1.0, 1.8])
    assert result.native_output_time_grid_sha256 != result.observation_time_grid_sha256
    assert result.source_identity_sha256 == "a" * 64
    assert result.output_identity_sha256 != result.observation_identity_sha256
    assert result.interpolation == PositionInterpolation.LINEAR_POSITION
    assert "align_native_positions_to_observations" in motion_matching_api.__all__


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_time_s", np.array([-0.01, 0.5, 0.9])),
        ("observation_time_s", np.array([0.1, 0.5, 1.01])),
    ],
)
def test_refuses_extrapolation_instead_of_clamping(
    field: str, value: np.ndarray
) -> None:
    values = _inputs()
    values[field] = value
    with pytest.raises(ValueError, match="extrapolation"):
        align_native_positions_to_observations(**values)


def test_requires_strictly_increasing_native_and_observation_clocks() -> None:
    for field, value in (
        ("native_output_time_s", np.array([0.0, 0.4, 0.4])),
        ("observation_time_s", np.array([0.1, 0.1, 0.9])),
    ):
        values = _inputs()
        values[field] = value
        with pytest.raises(ValueError, match="strictly increasing"):
            align_native_positions_to_observations(**values)


def test_preserves_missing_observation_mask_and_scores_on_observation_time() -> None:
    values = _inputs()
    target = np.asarray(values["observation_positions_m"]).copy()
    valid = np.asarray(values["observation_valid"]).copy()
    valid[1, 1] = False
    target[1, 1] = np.nan
    values["observation_positions_m"] = target
    values["observation_valid"] = valid

    result = align_native_positions_to_observations(**values)
    metrics = result.compute_replay_five_metrics()

    np.testing.assert_array_equal(result.observation_valid, valid)
    assert np.isnan(result.observation_positions_m[1, 1]).all()
    assert metrics.whole_rms_m == pytest.approx(1.0)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"native_frame_id": "model-local", "observation_frame_id": "world"}, "frame"),
        (
            {
                "native_timebase_id": "input-grid",
                "observation_timebase_id": "capture-clock",
            },
            "timebase",
        ),
        ({"source_identity_sha256": "not-a-digest"}, "SHA-256"),
    ],
)
def test_requires_declared_matching_frame_clock_and_source_identity(
    updates: dict[str, str], match: str
) -> None:
    values = _inputs() | updates
    with pytest.raises(ValueError, match=match):
        align_native_positions_to_observations(**values)


def test_rejects_state_quaternions_and_nonfinite_native_positions() -> None:
    values = _inputs()
    values["native_output_positions_m"] = np.zeros((3, 2, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="three-dimensional marker positions"):
        align_native_positions_to_observations(**values)

    values = _inputs()
    positions = np.asarray(values["native_output_positions_m"]).copy()
    positions[1, 0, 0] = np.inf
    values["native_output_positions_m"] = positions
    with pytest.raises(ValueError, match="finite"):
        align_native_positions_to_observations(**values)


def test_output_digest_changes_when_native_positions_change() -> None:
    values = _inputs()
    first = align_native_positions_to_observations(**values)
    changed = np.asarray(values["native_output_positions_m"]).copy()
    changed[1, 0, 0] += 0.01
    values["native_output_positions_m"] = changed
    second = align_native_positions_to_observations(**values)

    assert first.output_identity_sha256 != second.output_identity_sha256
    assert first.observation_time_grid_sha256 == second.observation_time_grid_sha256


def test_requires_matching_marker_order_and_declared_interpolation() -> None:
    values = _inputs()
    values["observation_marker_labels"] = ("WaistRight", "WaistLeft")
    with pytest.raises(ValueError, match="marker identities or order"):
        align_native_positions_to_observations(**values)

    values = _inputs()
    values["interpolation"] = "slerp_quaternion"
    with pytest.raises(ValueError, match="unsupported marker-position interpolation"):
        align_native_positions_to_observations(**values)


def test_observation_clock_change_does_not_rewrite_native_output_clock_identity() -> (
    None
):
    values = _inputs()
    first = align_native_positions_to_observations(**values)
    changed = dict(values)
    changed["observation_time_s"] = np.array([0.2, 0.6, 0.8])
    second = align_native_positions_to_observations(**changed)

    assert first.native_output_time_grid_sha256 == second.native_output_time_grid_sha256
    assert first.observation_time_grid_sha256 != second.observation_time_grid_sha256
    assert first.alignment_identity_sha256 != second.alignment_identity_sha256
    assert np.array_equal(second.observation_time_s, changed["observation_time_s"])
