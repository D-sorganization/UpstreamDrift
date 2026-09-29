"""Tests for src.shared.python.pendulum_simulator.dynamics_quantities (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.dynamics_quantities import (
    angular_impulse_series,
    angular_power_at,
    angular_power_series,
    angular_work_series,
    linear_power_at,
    linear_power_series,
)


class TestAngularPowerAt:
    def test_positive_torque_positive_velocity(self) -> None:
        assert angular_power_at(10.0, 2.0) == pytest.approx(20.0)

    def test_zero_torque_zero_power(self) -> None:
        assert angular_power_at(0.0, 5.0) == pytest.approx(0.0)

    def test_negative_velocity_negative_power(self) -> None:
        assert angular_power_at(5.0, -2.0) == pytest.approx(-10.0)

    def test_both_negative_positive_power(self) -> None:
        assert angular_power_at(-3.0, -4.0) == pytest.approx(12.0)


class TestLinearPowerAt:
    def test_aligned_vectors_positive_power(self) -> None:
        f = np.array([1.0, 0.0])
        v = np.array([2.0, 0.0])
        assert linear_power_at(f, v) == pytest.approx(2.0)

    def test_perpendicular_vectors_zero_power(self) -> None:
        f = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert linear_power_at(f, v) == pytest.approx(0.0)

    def test_anti_aligned_negative_power(self) -> None:
        f = np.array([1.0, 0.0])
        v = np.array([-3.0, 0.0])
        assert linear_power_at(f, v) == pytest.approx(-3.0)


class TestAngularPowerSeries:
    def test_constant_torque_and_velocity(self) -> None:
        torques = np.ones(5) * 4.0
        omegas = np.ones(5) * 2.0
        result = angular_power_series(torques, omegas)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, 8.0)

    def test_zero_torque_zero_power(self) -> None:
        torques = np.zeros(4)
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        result = angular_power_series(torques, omegas)
        np.testing.assert_allclose(result, 0.0)

    def test_dynamics_quantities_returns_ndarray(self) -> None:
        result = angular_power_series(np.ones(3), np.ones(3))
        assert isinstance(result, np.ndarray)


class TestAngularWorkSeries:
    def test_constant_power_cumulative(self) -> None:
        torques = np.ones(10) * 2.0
        omegas = np.ones(10) * 3.0
        time = np.linspace(0.0, 0.9, 10)
        result = angular_work_series(torques, omegas, time)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)

    def test_zero_torque_zero_work(self) -> None:
        torques = np.zeros(5)
        omegas = np.ones(5)
        time = np.linspace(0.0, 0.4, 5)
        result = angular_work_series(torques, omegas, time)
        np.testing.assert_allclose(result, 0.0)


class TestAngularImpulseSeries:
    def test_constant_torque_cumulative(self) -> None:
        torques = np.ones(5) * 10.0
        time = np.linspace(0.0, 0.4, 5)
        result = angular_impulse_series(torques, time)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_zero_torque_zero_impulse(self) -> None:
        time = np.linspace(0.0, 0.4, 5)
        result = angular_impulse_series(np.zeros(5), time)
        np.testing.assert_allclose(result, 0.0)


class TestLinearPowerSeries:
    def test_dynamics_quantities_returns_ndarray(self) -> None:
        forces = np.zeros((5, 2))
        forces[:, 0] = 1.0
        vels = np.zeros((5, 2))
        vels[:, 0] = 2.0
        result = linear_power_series(forces, vels)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, 2.0)
