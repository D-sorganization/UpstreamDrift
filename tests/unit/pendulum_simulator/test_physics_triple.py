"""Tests for src.shared.python.pendulum_simulator.physics_triple (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.physics_triple import (
    TriplePendulumParams,
    coriolis_vector,
    forward_kinematics,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
)


def _make_params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=1.0, m2=0.5, m3=0.3, L1=1.0, L2=0.5, L3=0.4)


_P = _make_params()


class TestTriplePendulumParams:
    def test_physics_triple_construction(self) -> None:
        p = _make_params()
        assert isinstance(p, TriplePendulumParams)

    def test_masses_stored(self) -> None:
        p = _make_params()
        assert p.m1 == pytest.approx(1.0)
        assert p.m2 == pytest.approx(0.5)
        assert p.m3 == pytest.approx(0.3)

    def test_lengths_stored(self) -> None:
        p = _make_params()
        assert pytest.approx(1.0) == p.L1
        assert pytest.approx(0.5) == p.L2
        assert pytest.approx(0.4) == p.L3

    def test_physics_triple_default_gravity(self) -> None:
        p = _make_params()
        assert p.g == pytest.approx(9.81)


class TestMassMatrix:
    def test_returns_3x3_array(self) -> None:
        M = mass_matrix(0.0, 0.0, _P)
        assert M.shape == (3, 3)

    def test_physics_triple_is_symmetric(self) -> None:
        M = mass_matrix(0.1, -0.2, _P)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_is_positive_definite(self) -> None:
        M = mass_matrix(0.0, 0.0, _P)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)

    def test_physics_triple_finite_values(self) -> None:
        M = mass_matrix(0.5, -0.3, _P)
        assert np.all(np.isfinite(M))

    def test_physics_triple_angle_dependence(self) -> None:
        M1 = mass_matrix(0.0, 0.0, _P)
        M2 = mass_matrix(0.5, 0.5, _P)
        assert not np.allclose(M1, M2)


class TestGravityVector:
    def test_returns_shape_3(self) -> None:
        G = gravity_vector(0.0, 0.0, 0.0, _P)
        assert G.shape == (3,)

    def test_physics_triple_finite_values(self) -> None:
        G = gravity_vector(0.3, -0.2, 0.1, _P)
        assert np.all(np.isfinite(G))

    def test_zero_gravity_zero_vector(self) -> None:
        p_no_g = TriplePendulumParams(
            m1=1.0, m2=0.5, m3=0.3, L1=1.0, L2=0.5, L3=0.4, g=0.0
        )
        G = gravity_vector(0.0, 0.0, 0.0, p_no_g)
        np.testing.assert_allclose(G, [0.0, 0.0, 0.0])


class TestCoriolisVector:
    def test_returns_shape_3(self) -> None:
        C = coriolis_vector(0.0, 0.0, 0.0, 0.0, 0.0, _P)
        assert C.shape == (3,)

    def test_physics_triple_finite_values(self) -> None:
        C = coriolis_vector(0.1, 0.2, 0.5, -0.3, 0.2, _P)
        assert np.all(np.isfinite(C))

    def test_physics_triple_zero_velocities_zero_coriolis(self) -> None:
        C = coriolis_vector(0.0, 0.0, 0.0, 0.0, 0.0, _P)
        np.testing.assert_allclose(C, [0.0, 0.0, 0.0], atol=1e-12)


class TestForwardKinematics:
    def test_physics_triple_returns_dict(self) -> None:
        result = forward_kinematics(0.0, 0.0, 0.0, _P)
        assert isinstance(result, dict)

    def test_physics_triple_has_expected_keys(self) -> None:
        result = forward_kinematics(0.0, 0.0, 0.0, _P)
        assert "hub" in result
        assert "shoulder" in result
        assert "wrist1" in result
        assert "wrist2" in result
        assert "tip" in result

    def test_physics_triple_finite_values(self) -> None:
        result = forward_kinematics(0.3, -0.2, 0.1, _P)
        for val in result.values():
            assert np.isfinite(val[0]) and np.isfinite(val[1])


class TestKineticEnergy:
    def test_zero_state_zero_ke(self) -> None:
        state = np.zeros(6)
        ke = kinetic_energy(state, _P)
        assert ke == pytest.approx(0.0)

    def test_ke_positive_for_nonzero_velocities(self) -> None:
        state = np.zeros(6)
        state[3] = 1.0  # dtheta1 = 1.0
        ke = kinetic_energy(state, _P)
        assert ke > 0.0

    def test_ke_finite(self) -> None:
        state = np.array([0.2, 0.1, -0.1, 0.5, 0.3, -0.2])
        ke = kinetic_energy(state, _P)
        assert np.isfinite(ke)
