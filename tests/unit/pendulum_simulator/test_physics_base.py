"""Tests for src.shared.python.pendulum_simulator.physics_base (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.physics_base import (
    clamp_torque_ndof,
    hermite_smoothstep,
    kinetic_energy_from_M,
    potential_energy_chain,
    total_energy_from_parts,
)


class TestKineticEnergyFromM:
    def test_identity_mass_matrix(self) -> None:
        M = np.eye(2)
        qdot = np.array([1.0, 2.0])
        # KE = 0.5 * qdot.T @ M @ qdot = 0.5 * (1+4) = 2.5
        assert kinetic_energy_from_M(M, qdot) == pytest.approx(2.5)

    def test_zero_velocity_zero_ke(self) -> None:
        M = np.eye(3)
        qdot = np.zeros(3)
        assert kinetic_energy_from_M(M, qdot) == pytest.approx(0.0)

    def test_scaled_mass_matrix(self) -> None:
        M = 2.0 * np.eye(2)
        qdot = np.array([1.0, 1.0])
        # KE = 0.5 * qdot @ 2I @ qdot = 0.5 * 2 * 2 = 2.0
        assert kinetic_energy_from_M(M, qdot) == pytest.approx(2.0)

    def test_ke_is_non_negative(self) -> None:
        M = np.eye(4)
        qdot = np.array([-1.0, 2.0, -3.0, 4.0])
        assert kinetic_energy_from_M(M, qdot) >= 0.0


class TestTotalEnergyFromParts:
    def test_sum_of_parts(self) -> None:
        assert total_energy_from_parts(5.0, 3.0) == pytest.approx(8.0)

    def test_zero_parts(self) -> None:
        assert total_energy_from_parts(0.0, 0.0) == pytest.approx(0.0)

    def test_negative_potential(self) -> None:
        assert total_energy_from_parts(10.0, -2.0) == pytest.approx(8.0)


class TestClampTorqueNdof:
    def test_within_limits_unchanged(self) -> None:
        tau = np.array([1.0, -1.0])
        limits = np.array([3.0, 3.0])
        result = clamp_torque_ndof(tau, limits)
        np.testing.assert_array_equal(result, tau)

    def test_exceeds_positive_limit_clamped(self) -> None:
        tau = np.array([5.0, 0.0])
        limits = np.array([3.0, 3.0])
        result = clamp_torque_ndof(tau, limits)
        assert result[0] == pytest.approx(3.0)

    def test_exceeds_negative_limit_clamped(self) -> None:
        tau = np.array([0.0, -5.0])
        limits = np.array([3.0, 3.0])
        result = clamp_torque_ndof(tau, limits)
        assert result[1] == pytest.approx(-3.0)

    def test_shape_preserved(self) -> None:
        tau = np.zeros(4)
        limits = np.ones(4) * 10.0
        result = clamp_torque_ndof(tau, limits)
        assert result.shape == (4,)


class TestHermiteSmoothstep:
    def test_at_zero_returns_zero(self) -> None:
        assert hermite_smoothstep(0.0) == pytest.approx(0.0)

    def test_at_one_returns_one(self) -> None:
        assert hermite_smoothstep(1.0) == pytest.approx(1.0)

    def test_at_half_returns_half(self) -> None:
        assert hermite_smoothstep(0.5) == pytest.approx(0.5)

    def test_monotonically_increasing(self) -> None:
        xs = np.linspace(0.0, 1.0, 20)
        values = [hermite_smoothstep(x) for x in xs]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


class TestPotentialEnergyChain:
    def test_physics_base_returns_float(self) -> None:
        lengths = np.array([1.0, 1.0])
        masses = np.array([1.0, 1.0])
        q = np.array([0.0, 0.0])  # hanging straight down
        pe = potential_energy_chain(q, lengths, masses, g=9.81)
        assert isinstance(float(pe), float)
