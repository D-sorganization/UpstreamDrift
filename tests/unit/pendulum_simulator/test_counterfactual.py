"""Tests for src.shared.python.pendulum_simulator.counterfactual (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.counterfactual import (
    zero_torque_joint_forces_double,
    zero_torque_joint_forces_triple,
)
from src.shared.python.pendulum_simulator.physics import PendulumParams
from src.shared.python.pendulum_simulator.physics_triple import TriplePendulumParams


def _make_double_params() -> PendulumParams:
    return PendulumParams(m1=1.0, m2=0.3, L1=1.0, L2=0.5)


def _make_triple_params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=1.0, m2=0.5, m3=0.3, L1=1.0, L2=0.5, L3=0.4)


class TestZeroTorqueJointForcesDouble:
    def test_counterfactual_returns_dict(self) -> None:
        state = np.zeros(4)
        result = zero_torque_joint_forces_double(state, _make_double_params())
        assert isinstance(result, dict)

    def test_counterfactual_has_expected_keys(self) -> None:
        state = np.zeros(4)
        result = zero_torque_joint_forces_double(state, _make_double_params())
        # Should have joint force entries
        assert len(result) > 0

    def test_finite_values_at_zero_state(self) -> None:
        state = np.zeros(4)
        result = zero_torque_joint_forces_double(state, _make_double_params())
        for key, val in result.items():
            fx, fy = val
            assert np.isfinite(fx), f"Non-finite fx at {key}"
            assert np.isfinite(fy), f"Non-finite fy at {key}"

    def test_wrong_state_shape_raises(self) -> None:
        with pytest.raises(AssertionError):
            zero_torque_joint_forces_double(np.zeros(6), _make_double_params())

    def test_non_finite_state_raises(self) -> None:
        state = np.array([np.nan, 0.0, 0.0, 0.0])
        with pytest.raises(AssertionError):
            zero_torque_joint_forces_double(state, _make_double_params())

    def test_nonzero_state_finite_output(self) -> None:
        state = np.array([0.3, -0.2, 0.5, -0.3])
        result = zero_torque_joint_forces_double(state, _make_double_params())
        for val in result.values():
            fx, fy = val
            assert np.isfinite(fx) and np.isfinite(fy)


class TestZeroTorqueJointForcesTriple:
    def test_counterfactual_returns_dict(self) -> None:
        state = np.zeros(6)
        result = zero_torque_joint_forces_triple(state, _make_triple_params())
        assert isinstance(result, dict)

    def test_counterfactual_has_expected_keys(self) -> None:
        state = np.zeros(6)
        result = zero_torque_joint_forces_triple(state, _make_triple_params())
        assert len(result) > 0

    def test_finite_values_at_zero_state(self) -> None:
        state = np.zeros(6)
        result = zero_torque_joint_forces_triple(state, _make_triple_params())
        for key, val in result.items():
            fx, fy = val
            assert np.isfinite(fx), f"Non-finite fx at {key}"
            assert np.isfinite(fy), f"Non-finite fy at {key}"

    def test_wrong_state_shape_raises(self) -> None:
        with pytest.raises(AssertionError):
            zero_torque_joint_forces_triple(np.zeros(4), _make_triple_params())

    def test_non_finite_state_raises(self) -> None:
        state = np.zeros(6)
        state[0] = np.nan
        with pytest.raises(AssertionError):
            zero_torque_joint_forces_triple(state, _make_triple_params())

    def test_nonzero_state_finite_output(self) -> None:
        state = np.array([0.2, -0.1, 0.3, 0.5, -0.3, 0.2])
        result = zero_torque_joint_forces_triple(state, _make_triple_params())
        for val in result.values():
            fx, fy = val
            assert np.isfinite(fx) and np.isfinite(fy)
