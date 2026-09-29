"""Tests for distinct TorqueTrajectory and MuscleActivationTrajectory types.

Issue #4667. Validates that torque and muscle-activation trajectories are
distinct Pydantic models with their own invariants — and that
``MotionMatchingResult`` accepts them as alternatives to a matched joint
trajectory.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.motion_pipeline.contracts import (
    MotionMatchingResult,
    MuscleActivationFrame,
    MuscleActivationTrajectory,
    TorqueFrame,
    TorqueTrajectory,
)

# -----------------------------------------------------------------------------
# TorqueTrajectory
# -----------------------------------------------------------------------------


def _make_torque_traj(
    taus: list[list[float]],
    times: list[float] | None = None,
    rig_joint_names: list[str] | None = None,
) -> TorqueTrajectory:
    if times is None:
        times = [float(i) * 0.01 for i in range(len(taus))]
    if rig_joint_names is None:
        rig_joint_names = ["j0", "j1"] if taus and len(taus[0]) == 2 else ["j0"]
    frames = [
        TorqueFrame(timestamp=t, tau=tau) for t, tau in zip(times, taus, strict=True)
    ]
    return TorqueTrajectory(frames=frames, rig_joint_names=rig_joint_names)


def test_torque_trajectory_valid_construction():
    traj = _make_torque_traj([[0.1, -0.2], [0.3, 0.4], [-1.0, 1.5]])
    assert traj.num_frames == 3
    assert traj.duration == pytest.approx(0.02)
    assert traj.rig_joint_names == ["j0", "j1"]


def test_torque_trajectory_rejects_nan_tau():
    with pytest.raises(ValueError, match="finite"):
        TorqueFrame(timestamp=0.0, tau=[float("nan"), 0.0])


def test_torque_trajectory_rejects_inf_tau():
    with pytest.raises(ValueError, match="finite"):
        TorqueFrame(timestamp=0.0, tau=[math.inf, 0.0])


def test_torque_trajectory_rejects_dim_mismatch():
    frames = [
        TorqueFrame(timestamp=0.0, tau=[0.1, 0.2]),
        TorqueFrame(timestamp=0.01, tau=[0.3]),  # length 1 != joint_names len 2
    ]
    with pytest.raises(ValueError, match="length"):
        TorqueTrajectory(frames=frames, rig_joint_names=["j0", "j1"])


def test_torque_trajectory_rejects_non_monotonic_time():
    with pytest.raises(ValueError, match="monotonic"):
        _make_torque_traj([[0.0, 0.0], [0.1, 0.1]], times=[0.01, 0.0])


def test_torque_trajectory_rejects_equal_timestamps():
    with pytest.raises(ValueError, match="monotonic"):
        _make_torque_traj([[0.0, 0.0], [0.1, 0.1]], times=[0.0, 0.0])


def test_torque_trajectory_rejects_empty_frames():
    with pytest.raises(ValueError, match="at least one frame"):
        TorqueTrajectory(frames=[], rig_joint_names=["j0"])


# -----------------------------------------------------------------------------
# MuscleActivationTrajectory
# -----------------------------------------------------------------------------


def _make_activation_traj(
    activations: list[list[float]],
    times: list[float] | None = None,
    muscle_names: list[str] | None = None,
) -> MuscleActivationTrajectory:
    if times is None:
        times = [float(i) * 0.01 for i in range(len(activations))]
    if muscle_names is None:
        muscle_names = (
            ["m0", "m1"] if activations and len(activations[0]) == 2 else ["m0"]
        )
    frames = [
        MuscleActivationFrame(timestamp=t, activations=a)
        for t, a in zip(times, activations, strict=True)
    ]
    return MuscleActivationTrajectory(frames=frames, muscle_names=muscle_names)


def test_muscle_activation_trajectory_valid_construction():
    traj = _make_activation_traj([[0.0, 0.5], [1.0, 0.25], [0.7, 0.1]])
    assert traj.num_frames == 3
    assert traj.muscle_names == ["m0", "m1"]


@pytest.mark.parametrize(
    "bad_value",
    [-0.1, 1.1, float("nan"), math.inf, -math.inf],
)
def test_muscle_activation_trajectory_rejects_out_of_range(bad_value: float):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MuscleActivationFrame(timestamp=0.0, activations=[bad_value, 0.5])


def test_muscle_activation_trajectory_rejects_dim_mismatch():
    frames = [
        MuscleActivationFrame(timestamp=0.0, activations=[0.1, 0.2]),
        MuscleActivationFrame(timestamp=0.01, activations=[0.3]),
    ]
    with pytest.raises(ValueError, match="length"):
        MuscleActivationTrajectory(frames=frames, muscle_names=["m0", "m1"])


def test_muscle_activation_trajectory_rejects_non_monotonic_time():
    with pytest.raises(ValueError, match="monotonic"):
        _make_activation_traj([[0.5, 0.5], [0.5, 0.5]], times=[0.01, 0.0])


def test_muscle_activation_trajectory_rejects_empty_frames():
    with pytest.raises(ValueError, match="at least one frame"):
        MuscleActivationTrajectory(frames=[], muscle_names=["m0"])


def test_muscle_activation_trajectory_accepts_boundary_values():
    traj = _make_activation_traj([[0.0, 1.0], [1.0, 0.0]])
    assert traj.num_frames == 2


# -----------------------------------------------------------------------------
# MotionMatchingResult — accepts torques OR activations
# -----------------------------------------------------------------------------


def test_motion_matching_result_accepts_torque_only():
    torques = _make_torque_traj([[0.1, 0.2], [0.3, 0.4]])
    res = MotionMatchingResult(
        request_id="r1",
        success=True,
        torques=torques,
    )
    assert res.torques is torques
    assert res.activations is None


def test_motion_matching_result_accepts_activation_only():
    acts = _make_activation_traj([[0.1, 0.5], [0.2, 0.6]])
    res = MotionMatchingResult(
        request_id="r2",
        success=True,
        activations=acts,
    )
    assert res.activations is acts
    assert res.torques is None


def test_motion_matching_result_accepts_both():
    torques = _make_torque_traj([[0.1, 0.2], [0.3, 0.4]])
    acts = _make_activation_traj([[0.1, 0.5], [0.2, 0.6]])
    res = MotionMatchingResult(
        request_id="r3",
        success=True,
        torques=torques,
        activations=acts,
    )
    assert res.torques is torques
    assert res.activations is acts


def test_motion_matching_result_rejects_when_successful_but_no_payload():
    """A success=True result with no matched_trajectory, torques, or
    activations is invalid."""
    with pytest.raises(ValueError, match="at least one"):
        MotionMatchingResult(request_id="r4", success=True)


def test_motion_matching_result_failed_no_payload_ok():
    """A failed result without payload is allowed (carries only message)."""
    res = MotionMatchingResult(request_id="r5", success=False, message="solver failed")
    assert res.success is False
    assert res.torques is None
    assert res.activations is None


# -----------------------------------------------------------------------------
# JSON round-trip
# -----------------------------------------------------------------------------


def test_torque_trajectory_json_round_trip():
    original = _make_torque_traj([[0.1, -0.2], [0.3, 0.4]])
    json_str = original.model_dump_json()
    restored = TorqueTrajectory.model_validate_json(json_str)
    assert restored.num_frames == original.num_frames
    assert restored.rig_joint_names == original.rig_joint_names
    for a, b in zip(original.frames, restored.frames, strict=True):
        assert a.timestamp == b.timestamp
        assert a.tau == b.tau


def test_muscle_activation_trajectory_json_round_trip():
    original = _make_activation_traj([[0.1, 0.5], [0.2, 0.6]])
    json_str = original.model_dump_json()
    restored = MuscleActivationTrajectory.model_validate_json(json_str)
    assert restored.num_frames == original.num_frames
    assert restored.muscle_names == original.muscle_names
    for a, b in zip(original.frames, restored.frames, strict=True):
        assert a.timestamp == b.timestamp
        assert a.activations == b.activations
