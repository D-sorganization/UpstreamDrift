"""Tests for src.shared.python.pendulum_simulator.golfer_kinematics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.golfer_kinematics import forward_kinematics
from src.shared.python.pendulum_simulator.physics_golfer import GolferParams


def _make_params(**kwargs) -> GolferParams:
    defaults = {
        "m_hub": 10.0,
        "m_r_upper": 2.0,
        "m_r_fore": 1.5,
        "m_l_upper": 2.0,
        "m_l_fore": 1.5,
        "m_club": 0.4,
        "L_hub": 0.2,
        "L_r_upper": 0.3,
        "L_r_fore": 0.25,
        "L_l_upper": 0.3,
        "L_l_fore": 0.25,
        "L_club": 1.0,
        "d_rs": 0.2,
        "d_ls": 0.2,
        "grip_right": 0.05,
        "grip_left": 0.15,
    }
    defaults.update(kwargs)
    return GolferParams(**defaults)


_DEFAULT_P = _make_params()
_ZERO_Q = np.zeros(8)


class TestForwardKinematicsReturnKeys:
    def test_golfer_kinematics_returns_dict(self) -> None:
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        assert isinstance(result, dict)

    def test_has_hub_key(self) -> None:
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        assert "hub" in result

    def test_has_arm_keys(self) -> None:
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        for key in ("rs", "re", "rh", "ls", "le", "lh"):
            assert key in result, f"Missing key: {key}"

    def test_has_club_keys(self) -> None:
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        for key in ("club_base", "club_tip"):
            assert key in result, f"Missing key: {key}"

    def test_values_are_2_tuples(self) -> None:
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        for key, val in result.items():
            assert len(val) == 2, f"Key {key!r} should be length-2 tuple"


class TestForwardKinematicsZeroAngles:
    def test_hub_at_correct_height(self) -> None:
        # At q=0, hub extends straight up: y = L_hub
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        hub_y = result["hub"][1]
        assert hub_y == pytest.approx(_DEFAULT_P.L_hub)

    def test_hub_x_zero_at_zero_angle(self) -> None:
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        assert result["hub"][0] == pytest.approx(0.0, abs=1e-10)

    def test_right_hand_further_from_origin_than_right_elbow(self) -> None:
        # With zero angles, arms hang down — RH should be lower than RE
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        rh_y = result["rh"][1]
        re_y = result["re"][1]
        assert rh_y < re_y

    def test_symmetry_left_right_at_zero(self) -> None:
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        # At zero angles with equal parameters, left and right should be mirrored
        rs_x = result["rs"][0]
        ls_x = result["ls"][0]
        assert rs_x == pytest.approx(-ls_x, abs=1e-10)

    def test_club_tip_further_than_club_base(self) -> None:
        result = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        tip = result["club_tip"]
        base = result["club_base"]
        tip_dist = np.hypot(tip[0], tip[1])
        base_dist = np.hypot(base[0], base[1])
        assert tip_dist != base_dist  # not coincident


class TestForwardKinematicsRotation:
    def test_nonzero_hub_angle_changes_hub_position(self) -> None:
        q_rotated = _ZERO_Q.copy()
        q_rotated[0] = np.pi / 4
        result_zero = forward_kinematics(_ZERO_Q, _DEFAULT_P)
        result_rot = forward_kinematics(q_rotated, _DEFAULT_P)
        assert result_zero["hub"] != pytest.approx(result_rot["hub"])

    def test_all_positions_finite(self) -> None:
        q_test = np.array([0.1, 0.2, -0.1, 0.05, 0.15, -0.05, 0.1, 0.3])
        result = forward_kinematics(q_test, _DEFAULT_P)
        for key, val in result.items():
            assert np.all(np.isfinite(val)), f"Non-finite value for key {key!r}: {val}"
