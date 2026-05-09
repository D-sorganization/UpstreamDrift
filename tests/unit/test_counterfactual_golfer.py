"""Tests for pendulum_simulator.counterfactual_golfer (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.counterfactual_golfer import (
    zero_torque_accelerations,
    zero_torque_joint_forces,
)
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams


def _make_params() -> GolferParams:
    return GolferParams(
        m_hub=0.01,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.3,
        L_hub=0.05,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.2,
        grip_left=0.25,
    )


class TestZeroTorqueAccelerations:
    def test_counterfactual_golfer_returns_ndarray(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        acc = zero_torque_accelerations(state, params)
        assert isinstance(acc, np.ndarray)

    def test_counterfactual_golfer_output_shape(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        acc = zero_torque_accelerations(state, params)
        assert acc.shape == (N_DOF,)

    def test_wrong_state_shape_raises(self) -> None:
        params = _make_params()
        state = np.zeros(4)
        with pytest.raises((AssertionError, ValueError)):
            zero_torque_accelerations(state, params)


class TestZeroTorqueJointForces:
    def test_counterfactual_golfer_returns_dict(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        forces = zero_torque_joint_forces(state, params)
        assert isinstance(forces, dict)

    def test_forces_have_tuple_values(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        forces = zero_torque_joint_forces(state, params)
        for val in forces.values():
            assert len(val) == 2
