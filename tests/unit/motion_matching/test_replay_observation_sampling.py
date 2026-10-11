"""F09a boundaries between native output clocks and measured clocks."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from src.shared.python import motion_matching as motion_matching_api
from src.shared.python.motion_matching import (
    NativeMarkerPositionOutput,
    ObservedMarkerPositions,
    PositionInterpolation,
    ReplayObservationAlignment,
    align_native_positions_to_observations,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Inputs:
    native_output: NativeMarkerPositionOutput
    observations: ObservedMarkerPositions
    interpolation: PositionInterpolation
    source_identity_sha256: str

    def align(self) -> ReplayObservationAlignment:
        return align_native_positions_to_observations(
            self.native_output,
            self.observations,
            interpolation=self.interpolation,
            source_identity_sha256=self.source_identity_sha256,
        )


def _inputs() -> _Inputs:
    native_time = np.array([0.0, 0.4, 1.0])
    observation_time = np.array([0.1, 0.5, 0.9])
    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = native_time
    positions[:, 1, 1] = 2.0 * native_time
    target = np.zeros((3, 2, 3), dtype=np.float64)
    target[:, 0, 0] = observation_time + 1.0
    target[:, 1, 1] = 2.0 * observation_time + 1.0
    valid = np.ones((3, 2), dtype=bool)
    return _Inputs(
        native_output=NativeMarkerPositionOutput(
            time_s=native_time,
            positions_m=positions,
            marker_labels=("WaistLeft", "WaistRight"),
            frame_id="world-z-up-v1",
            timebase_id="simulation-relative-v1",
        ),
        observations=ObservedMarkerPositions(
            time_s=observation_time,
            positions_m=target,
            valid=valid,
            marker_labels=("WaistLeft", "WaistRight"),
            frame_id="world-z-up-v1",
            timebase_id="simulation-relative-v1",
        ),
        interpolation=PositionInterpolation.LINEAR_POSITION,
        source_identity_sha256="a" * 64,
    )


def test_aligns_positions_to_distinct_observation_clock_without_losing_it() -> None:
    result = _inputs().align()

    np.testing.assert_array_equal(result.observation_time_s, [0.1, 0.5, 0.9])
    np.testing.assert_allclose(result.predicted_positions_m[:, 0, 0], [0.1, 0.5, 0.9])
    np.testing.assert_allclose(result.predicted_positions_m[:, 1, 1], [0.2, 1.0, 1.8])
    assert result.native_output_time_grid_sha256 != result.observation_time_grid_sha256
    assert result.source_identity_sha256 == "a" * 64
    assert result.output_identity_sha256 != result.observation_identity_sha256
    assert result.interpolation == PositionInterpolation.LINEAR_POSITION
    assert "align_native_positions_to_observations" in motion_matching_api.__all__
    assert "NativeMarkerPositionOutput" in motion_matching_api.__all__
    assert "ObservedMarkerPositions" in motion_matching_api.__all__


@pytest.mark.parametrize(
    "value",
    [
        np.array([-0.01, 0.5, 0.9]),
        np.array([0.1, 0.5, 1.01]),
    ],
)
def test_refuses_extrapolation_instead_of_clamping(
    value: np.ndarray,
) -> None:
    values = _inputs()
    values = replace(
        values,
        observations=replace(values.observations, time_s=value),
    )
    with pytest.raises(ValueError, match="extrapolation"):
        values.align()


def test_requires_strictly_increasing_native_and_observation_clocks() -> None:
    for native_time, observation_time in (
        (np.array([0.0, 0.4, 0.4]), np.array([0.1, 0.5, 0.9])),
        (np.array([0.0, 0.4, 1.0]), np.array([0.1, 0.1, 0.9])),
    ):
        values = _inputs()
        values = replace(
            values,
            native_output=replace(values.native_output, time_s=native_time),
            observations=replace(values.observations, time_s=observation_time),
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            values.align()


def test_preserves_missing_observation_mask_and_scores_on_observation_time() -> None:
    values = _inputs()
    target = np.asarray(values.observations.positions_m).copy()
    valid = np.asarray(values.observations.valid).copy()
    valid[1, 1] = False
    target[1, 1] = np.nan
    values = replace(
        values,
        observations=replace(values.observations, positions_m=target, valid=valid),
    )

    result = values.align()
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
    values = _inputs()
    if "native_frame_id" in updates:
        values = replace(
            values,
            native_output=replace(
                values.native_output, frame_id=str(updates["native_frame_id"])
            ),
        )
    if "observation_frame_id" in updates:
        values = replace(
            values,
            observations=replace(
                values.observations, frame_id=str(updates["observation_frame_id"])
            ),
        )
    if "native_timebase_id" in updates:
        values = replace(
            values,
            native_output=replace(
                values.native_output, timebase_id=str(updates["native_timebase_id"])
            ),
        )
    if "observation_timebase_id" in updates:
        values = replace(
            values,
            observations=replace(
                values.observations,
                timebase_id=str(updates["observation_timebase_id"]),
            ),
        )
    if "source_identity_sha256" in updates:
        values = replace(
            values, source_identity_sha256=str(updates["source_identity_sha256"])
        )
    with pytest.raises(ValueError, match=match):
        values.align()


def test_rejects_state_quaternions_and_nonfinite_native_positions() -> None:
    values = _inputs()
    values = replace(
        values,
        native_output=replace(
            values.native_output,
            positions_m=np.zeros((3, 2, 4), dtype=np.float64),
        ),
    )
    with pytest.raises(ValueError, match="three-dimensional marker positions"):
        values.align()

    values = _inputs()
    positions = np.asarray(values.native_output.positions_m).copy()
    positions[1, 0, 0] = np.inf
    values = replace(
        values,
        native_output=replace(values.native_output, positions_m=positions),
    )
    with pytest.raises(ValueError, match="finite"):
        values.align()


def test_output_digest_changes_when_native_positions_change() -> None:
    values = _inputs()
    first = values.align()
    changed = np.asarray(values.native_output.positions_m).copy()
    changed[1, 0, 0] += 0.01
    values = replace(
        values,
        native_output=replace(values.native_output, positions_m=changed),
    )
    second = values.align()

    assert first.output_identity_sha256 != second.output_identity_sha256
    assert first.observation_time_grid_sha256 == second.observation_time_grid_sha256


def test_requires_matching_marker_order_and_declared_interpolation() -> None:
    values = _inputs()
    values = replace(
        values,
        observations=replace(
            values.observations, marker_labels=("WaistRight", "WaistLeft")
        ),
    )
    with pytest.raises(ValueError, match="marker identities or order"):
        values.align()

    values = _inputs()
    values = replace(values, interpolation="slerp_quaternion")
    with pytest.raises(ValueError, match="unsupported marker-position interpolation"):
        values.align()


def test_observation_clock_change_does_not_rewrite_native_output_clock_identity() -> (
    None
):
    values = _inputs()
    first = values.align()
    changed = replace(
        values,
        observations=replace(
            values.observations,
            time_s=np.array([0.2, 0.6, 0.8]),
        ),
    )
    second = changed.align()

    assert first.native_output_time_grid_sha256 == second.native_output_time_grid_sha256
    assert first.observation_time_grid_sha256 != second.observation_time_grid_sha256
    assert first.alignment_identity_sha256 != second.alignment_identity_sha256
    assert np.array_equal(second.observation_time_s, changed.observations.time_s)
