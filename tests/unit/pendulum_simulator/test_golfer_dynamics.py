"""Tests for src.shared.python.pendulum_simulator.golfer_dynamics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.golfer_dynamics import (
    analytical_coriolis,
    analytical_fk_jacobians,
    analytical_gravity_vector,
    analytical_mass_matrix,
    kinetic_energy,
    potential_energy,
    potential_energy_from_q,
    total_energy,
)
from src.shared.python.pendulum_simulator.physics_golfer import (
    N_DOF,
    GolferParams,
    State,
)


def _make_params() -> GolferParams:
    return GolferParams(
        m_hub=10.0,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.4,
        L_hub=0.2,
        L_r_upper=0.3,
        L_r_fore=0.25,
        L_l_upper=0.3,
        L_l_fore=0.25,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.05,
        grip_left=0.15,
    )


_P = _make_params()
_Q = np.zeros(N_DOF)
_QDOT = np.zeros(N_DOF)


class TestAnalyticalMassMatrix:
    def test_returns_ndof_x_ndof(self) -> None:
        M = analytical_mass_matrix(_Q, _P)
        assert M.shape == (N_DOF, N_DOF)

    def test_golfer_dynamics_is_symmetric(self) -> None:
        M = analytical_mass_matrix(_Q, _P)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_is_positive_semidefinite(self) -> None:
        M = analytical_mass_matrix(_Q, _P)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues >= -1e-10)

    def test_golfer_dynamics_finite_values(self) -> None:
        M = analytical_mass_matrix(_Q, _P)
        assert np.all(np.isfinite(M))

    def test_nonzero_angles(self) -> None:
        q = np.zeros(N_DOF)
        q[0] = 0.3
        q[1] = 0.2
        M = analytical_mass_matrix(q, _P)
        assert M.shape == (N_DOF, N_DOF)
        assert np.all(np.isfinite(M))

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            analytical_mass_matrix([0.0] * N_DOF, _P)  # type: ignore[arg-type]

    def test_golfer_dynamics_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            analytical_mass_matrix(np.zeros(3), _P)


class TestAnalyticalGravityVector:
    def test_returns_shape_ndof(self) -> None:
        G = analytical_gravity_vector(_Q, _P)
        assert G.shape == (N_DOF,)

    def test_golfer_dynamics_finite_values(self) -> None:
        G = analytical_gravity_vector(_Q, _P)
        assert np.all(np.isfinite(G))

    def test_zero_gravity_zero_vector(self) -> None:
        p_no_g = GolferParams(
            m_hub=10.0,
            m_r_upper=2.0,
            m_r_fore=1.5,
            m_l_upper=2.0,
            m_l_fore=1.5,
            m_club=0.4,
            L_hub=0.2,
            L_r_upper=0.3,
            L_r_fore=0.25,
            L_l_upper=0.3,
            L_l_fore=0.25,
            L_club=1.0,
            d_rs=0.2,
            d_ls=0.2,
            grip_right=0.05,
            grip_left=0.15,
            g=0.0,
        )
        G = analytical_gravity_vector(_Q, p_no_g)
        np.testing.assert_allclose(G, np.zeros(N_DOF), atol=1e-12)

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            analytical_gravity_vector("bad", _P)  # type: ignore[arg-type]


class TestAnalyticalCoriolis:
    def test_returns_shape_ndof(self) -> None:
        C = analytical_coriolis(_Q, _QDOT, _P)
        assert C.shape == (N_DOF,)

    def test_golfer_dynamics_finite_values(self) -> None:
        C = analytical_coriolis(_Q, _QDOT, _P)
        assert np.all(np.isfinite(C))

    def test_golfer_dynamics_zero_velocities_zero_coriolis(self) -> None:
        C = analytical_coriolis(_Q, _QDOT, _P)
        np.testing.assert_allclose(C, np.zeros(N_DOF), atol=1e-10)

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            analytical_coriolis("bad", _QDOT, _P)  # type: ignore[arg-type]


class TestAnalyticalFkJacobians:
    _EXPECTED_KEYS = {"rh", "lh", "club_tip", "re", "le", "hub", "club_com", "rs", "ls"}

    def test_golfer_dynamics_returns_dict(self) -> None:
        result = analytical_fk_jacobians(_Q, _P)
        assert isinstance(result, dict)

    def test_golfer_dynamics_has_expected_keys(self) -> None:
        result = analytical_fk_jacobians(_Q, _P)
        # Verify all standard keys present
        for key in {"rh", "lh", "club_tip", "re", "le", "hub"}:
            assert key in result, f"Missing key: {key}"

    def test_shapes_are_2_by_ndof(self) -> None:
        result = analytical_fk_jacobians(_Q, _P)
        for key, J in result.items():
            assert J.shape == (
                2,
                N_DOF,
            ), f"Key {key}: expected (2, {N_DOF}), got {J.shape}"

    def test_golfer_dynamics_finite_values(self) -> None:
        result = analytical_fk_jacobians(_Q, _P)
        for key, J in result.items():
            assert np.all(np.isfinite(J)), f"Non-finite in Jacobian for {key}"


class TestKineticEnergy:
    def test_zero_velocities_zero_ke(self) -> None:
        ke = kinetic_energy(_Q, _QDOT, _P)
        assert ke == pytest.approx(0.0)

    def test_positive_for_nonzero_velocity(self) -> None:
        qdot = np.zeros(N_DOF)
        qdot[0] = 1.0
        ke = kinetic_energy(_Q, qdot, _P)
        assert ke > 0.0

    def test_finite(self) -> None:
        q = np.zeros(N_DOF)
        q[0] = 0.2
        qdot = np.zeros(N_DOF)
        qdot[0] = 0.5
        ke = kinetic_energy(q, qdot, _P)
        assert np.isfinite(ke)

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            kinetic_energy("bad", _QDOT, _P)  # type: ignore[arg-type]


class TestPotentialEnergyFromQ:
    def test_finite(self) -> None:
        pe = potential_energy_from_q(_Q, _P)
        assert np.isfinite(pe)

    def test_golfer_dynamics_returns_float(self) -> None:
        pe = potential_energy_from_q(_Q, _P)
        assert isinstance(pe, float)

    def test_golfer_dynamics_angle_dependence(self) -> None:
        q2 = np.zeros(N_DOF)
        q2[0] = 1.0
        pe1 = potential_energy_from_q(_Q, _P)
        pe2 = potential_energy_from_q(q2, _P)
        assert pe1 != pytest.approx(pe2)

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            potential_energy_from_q("bad", _P)  # type: ignore[arg-type]

    def test_golfer_dynamics_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            potential_energy_from_q(np.zeros(2), _P)


class TestPotentialEnergy:
    def test_finite(self) -> None:
        state: State = np.zeros(2 * N_DOF)
        pe = potential_energy(state, _P)
        assert np.isfinite(pe)


class TestTotalEnergy:
    def test_equals_ke_plus_pe(self) -> None:
        state: State = np.zeros(2 * N_DOF)
        state[0] = 0.2
        state[N_DOF] = 0.5  # qdot[0] = 0.5
        te = total_energy(state, _P)
        q = state[:N_DOF]
        qdot = state[N_DOF:]
        ke = kinetic_energy(q, qdot, _P)
        pe = potential_energy(state, _P)
        assert te == pytest.approx(ke + pe)

    def test_finite(self) -> None:
        state: State = np.zeros(2 * N_DOF)
        assert np.isfinite(total_energy(state, _P))
