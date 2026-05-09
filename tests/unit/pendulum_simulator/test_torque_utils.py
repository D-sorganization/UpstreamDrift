"""Tests for src.shared.python.pendulum_simulator.torque_utils (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.pendulum_simulator.torque_utils import make_polynomial_torque


class TestMakePolynomialTorque:
    def test_constant_torque_single_joint(self) -> None:
        tf = make_polynomial_torque([5.0])
        result = tf(0.0)
        assert len(result) == 1
        assert result[0] == pytest.approx(5.0)

    def test_constant_torque_at_nonzero_time(self) -> None:
        tf = make_polynomial_torque([7.0])
        assert tf(1.0)[0] == pytest.approx(7.0)
        assert tf(2.5)[0] == pytest.approx(7.0)

    def test_linear_polynomial(self) -> None:
        # c0 + c1*t: [3.0, 2.0] → tau(1.0) = 3.0 + 2.0*1.0 = 5.0
        tf = make_polynomial_torque([3.0, 2.0])
        assert tf(1.0)[0] == pytest.approx(5.0)
        assert tf(0.0)[0] == pytest.approx(3.0)

    def test_quadratic_polynomial(self) -> None:
        # c0 + c1*t + c2*t^2: [1.0, 0.0, 1.0] → tau(2.0) = 1 + 0 + 4 = 5.0
        tf = make_polynomial_torque([1.0, 0.0, 1.0])
        assert tf(2.0)[0] == pytest.approx(5.0)

    def test_two_joints_returns_tuple_len_2(self) -> None:
        tf = make_polynomial_torque([5.0], [3.0])
        result = tf(1.0)
        assert len(result) == 2

    def test_two_joints_values(self) -> None:
        tf = make_polynomial_torque([5.0], [3.0])
        tau_shoulder, tau_wrist = tf(0.0)
        assert tau_shoulder == pytest.approx(5.0)
        assert tau_wrist == pytest.approx(3.0)

    def test_three_joints(self) -> None:
        tf = make_polynomial_torque([1.0], [2.0], [3.0])
        result = tf(0.0)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)

    def test_zero_torque(self) -> None:
        tf = make_polynomial_torque([0.0])
        assert tf(5.0)[0] == pytest.approx(0.0)

    def test_torque_utils_returns_tuple(self) -> None:
        tf = make_polynomial_torque([1.0])
        assert isinstance(tf(0.0), tuple)

    def test_different_times_give_different_values(self) -> None:
        tf = make_polynomial_torque([0.0, 1.0])  # tau = t
        assert tf(1.0)[0] == pytest.approx(1.0)
        assert tf(2.0)[0] == pytest.approx(2.0)
        assert tf(3.0)[0] == pytest.approx(3.0)

    def test_no_joints_raises(self) -> None:
        with pytest.raises((AssertionError, TypeError)):
            make_polynomial_torque()

    def test_negative_torque(self) -> None:
        tf = make_polynomial_torque([-5.0])
        assert tf(0.0)[0] == pytest.approx(-5.0)
