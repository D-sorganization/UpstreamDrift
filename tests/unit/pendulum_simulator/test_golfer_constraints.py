"""Tests for src.shared.python.pendulum_simulator.golfer_constraints (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.golfer_constraints import (
    analytical_constraint_jacobian,
    constraint_vector,
    numerical_constraint_jacobian,
)
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams


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


class TestConstraintVector:
    def test_golfer_constraints_returns_ndarray(self) -> None:
        phi = constraint_vector(_Q, _P)
        assert isinstance(phi, np.ndarray)

    def test_shape_4(self) -> None:
        phi = constraint_vector(_Q, _P)
        assert phi.shape == (4,)

    def test_golfer_constraints_finite_values(self) -> None:
        phi = constraint_vector(_Q, _P)
        assert np.all(np.isfinite(phi))

    def test_wrong_type_q_raises(self) -> None:
        with pytest.raises(TypeError):
            constraint_vector([0.0] * N_DOF, _P)  # type: ignore[arg-type]

    def test_wrong_type_p_raises(self) -> None:
        with pytest.raises(TypeError):
            constraint_vector(_Q, "not_a_params")  # type: ignore[arg-type]

    def test_q_too_short_raises(self) -> None:
        with pytest.raises(ValueError):
            constraint_vector(np.zeros(3), _P)

    def test_longer_q_accepted(self) -> None:
        q_long = np.zeros(N_DOF + 4)
        phi = constraint_vector(q_long, _P)
        assert phi.shape == (4,)

    def test_different_angles_give_different_constraint(self) -> None:
        q1 = np.zeros(N_DOF)
        q2 = np.zeros(N_DOF)
        q2[0] = 0.5  # change hub angle
        phi1 = constraint_vector(q1, _P)
        phi2 = constraint_vector(q2, _P)
        assert not np.allclose(phi1, phi2)


class TestNumericalConstraintJacobian:
    def test_golfer_constraints_returns_ndarray(self) -> None:
        J = numerical_constraint_jacobian(_Q, _P)
        assert isinstance(J, np.ndarray)

    def test_shape_4_by_ndof(self) -> None:
        J = numerical_constraint_jacobian(_Q, _P)
        assert J.shape == (4, N_DOF)

    def test_golfer_constraints_finite_values(self) -> None:
        J = numerical_constraint_jacobian(_Q, _P)
        assert np.all(np.isfinite(J))

    def test_wrong_type_q_raises(self) -> None:
        with pytest.raises(TypeError):
            numerical_constraint_jacobian([0.0] * N_DOF, _P)  # type: ignore[arg-type]

    def test_wrong_type_p_raises(self) -> None:
        with pytest.raises(TypeError):
            numerical_constraint_jacobian(_Q, "not_a_params")  # type: ignore[arg-type]


class TestAnalyticalConstraintJacobian:
    def test_golfer_constraints_returns_ndarray(self) -> None:
        J = analytical_constraint_jacobian(_Q, _P)
        assert isinstance(J, np.ndarray)

    def test_shape_4_by_ndof(self) -> None:
        J = analytical_constraint_jacobian(_Q, _P)
        assert J.shape == (4, N_DOF)

    def test_golfer_constraints_finite_values(self) -> None:
        J = analytical_constraint_jacobian(_Q, _P)
        assert np.all(np.isfinite(J))

    def test_close_to_numerical_jacobian(self) -> None:
        J_num = numerical_constraint_jacobian(_Q, _P)
        J_ana = analytical_constraint_jacobian(_Q, _P)
        np.testing.assert_allclose(J_ana, J_num, atol=1e-4)
