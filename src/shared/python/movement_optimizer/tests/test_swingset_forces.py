# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for swingset force/torque estimates (models/swingset_forces.py)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from movement_optimizer.models import (
    SwingForceField,
    SwingForceHistory,
    swing_force_field,
    swing_force_history,
)
from movement_optimizer.models.swingset import (
    DEFAULT_POLICY_DT_S,
    SWING_POLICY_JOINT_NAMES,
    SwingPose,
    SwingRollout,
    SwingSetConfig,
    SwingSetState,
    estimate_swingset_joint_torques,
    heuristic_pumping_policy,
    simulate_swingset,
)

_STEPS = 30


def _make_rollout() -> tuple[SwingSetConfig, SwingRollout]:
    config = SwingSetConfig()
    start = SwingSetState(pose=SwingPose(swing_angle_rad=0.1))
    rollout = simulate_swingset(
        config, start, _STEPS, DEFAULT_POLICY_DT_S, heuristic_pumping_policy
    )
    return config, rollout


def _total_mass(config: SwingSetConfig) -> float:
    return config.seat_mass_kg + config.rider_mass_kg


def test_swing_force_field_structure() -> None:
    config, rollout = _make_rollout()
    field = swing_force_field(config, rollout, DEFAULT_POLICY_DT_S, frame_index=0)
    assert isinstance(field, SwingForceField)
    assert field.com_m.shape == (2,)
    assert field.gravity_n.shape == (2,)
    assert field.chain_tension_n.shape == (2,)
    assert field.joint_torque_nm.shape == (len(SWING_POLICY_JOINT_NAMES),)
    assert set(field.joint_points_m) == set(SWING_POLICY_JOINT_NAMES)
    for point in field.joint_points_m.values():
        assert point.shape == (2,)


def test_swing_force_field_gravity_points_down_with_weight_magnitude() -> None:
    config, rollout = _make_rollout()
    field = swing_force_field(config, rollout, DEFAULT_POLICY_DT_S, frame_index=5)
    expected = _total_mass(config) * config.gravity_m_s2
    assert field.gravity_n[0] == pytest.approx(0.0)
    assert field.gravity_n[1] == pytest.approx(expected)


@pytest.mark.parametrize("bad_index", [-1, _STEPS + 5, 9999])
def test_swing_force_field_rejects_out_of_range_index(bad_index: int) -> None:
    config, rollout = _make_rollout()
    with pytest.raises(ValueError, match="frame_index"):
        swing_force_field(config, rollout, DEFAULT_POLICY_DT_S, bad_index)


def test_swing_force_field_rejects_nonpositive_dt() -> None:
    config, rollout = _make_rollout()
    with pytest.raises(ValueError, match="dt_s"):
        swing_force_field(config, rollout, 0.0, 0)


def test_swing_force_history_shapes() -> None:
    config, rollout = _make_rollout()
    history = swing_force_history(config, rollout, DEFAULT_POLICY_DT_S)
    assert isinstance(history, SwingForceHistory)
    njoints = len(SWING_POLICY_JOINT_NAMES)
    assert history.time_s.shape == (_STEPS,)
    assert history.joint_torque_nm.shape == (_STEPS, njoints)
    assert history.joint_power_w.shape == (_STEPS, njoints)
    assert history.swing_angle_rad.shape == (_STEPS,)
    assert history.com_height_m.shape == (_STEPS,)
    assert history.com_path_m.shape == (_STEPS, 2)
    assert history.energy_j.shape == (_STEPS,)


def test_swing_force_history_power_is_torque_times_rate() -> None:
    config, rollout = _make_rollout()
    history = swing_force_history(config, rollout, DEFAULT_POLICY_DT_S)
    torques = estimate_swingset_joint_torques(config, rollout, DEFAULT_POLICY_DT_S)
    assert np.allclose(history.joint_power_w, torques * rollout.controls[:_STEPS])


def test_swing_force_history_energy_is_nonnegative() -> None:
    config, rollout = _make_rollout()
    history = swing_force_history(config, rollout, DEFAULT_POLICY_DT_S)
    assert np.all(history.energy_j >= 0.0)
    assert np.all(np.isfinite(history.energy_j))


def test_swing_force_history_rejects_nonpositive_dt() -> None:
    config, rollout = _make_rollout()
    with pytest.raises(ValueError, match="dt_s"):
        swing_force_history(config, rollout, -0.1)


def test_swing_force_history_rejects_bad_control_shape() -> None:
    config, rollout = _make_rollout()
    bad = dataclasses.replace(rollout, controls=np.zeros((3, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        swing_force_history(config, bad, DEFAULT_POLICY_DT_S)
