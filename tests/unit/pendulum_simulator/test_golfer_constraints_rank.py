"""Rank and sign characterisation of ``golfer_constraints`` (NSX-12, #12557).

Documents what the legacy loop-closure module actually delivers so no NSX
algorithm consumes it blindly.  The findings (verified here, not fixed here):

* ``Phi`` has 4 rows but the analytic Jacobian has rank 3, not 4;
* the two wrist-angle columns are identically zero (the hand "joints" never
  enter ``lh``/``rh``, which are forearm end points);
* ``forward_kinematics`` flips the sign of the y component of every club
  point (``rh_y + g*dy`` where ``dy = -cos``), while ``constraint_vector``
  rows 2-3 use the correct direction ``(sin, -cos)``.  The two agree only when
  ``cos(theta_club) = 0``, so every closed solution has a club angle of
  ``+-pi/2``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from src.shared.python.pendulum_simulator.golfer_constraints import (
    analytical_constraint_jacobian,
    constraint_vector,
    numerical_constraint_jacobian,
)
from src.shared.python.pendulum_simulator.golfer_kinematics import forward_kinematics
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams

pytestmark = pytest.mark.unit

WRIST_COLUMNS = (3, 6)


def _params() -> GolferParams:
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


def _closed_solutions(count: int = 12) -> list[np.ndarray]:
    p, rng = _params(), np.random.default_rng(0)
    out: list[np.ndarray] = []
    while len(out) < count:
        res = least_squares(
            lambda x: constraint_vector(x, p),
            rng.uniform(-3.0, 3.0, N_DOF),
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        if res.cost < 1e-20:
            out.append(res.x)
    return out


def test_jacobian_matches_finite_difference() -> None:
    q = np.random.default_rng(1).uniform(-1.0, 1.0, N_DOF)
    ana = analytical_constraint_jacobian(q, _params())
    num = numerical_constraint_jacobian(q, _params())
    assert np.max(np.abs(ana - num)) < 1e-6


def test_four_rows_but_rank_three_not_four() -> None:
    q = np.random.default_rng(1).uniform(-1.0, 1.0, N_DOF)
    jac = analytical_constraint_jacobian(q, _params())
    assert jac.shape == (4, N_DOF)
    assert np.linalg.matrix_rank(jac, tol=1e-8) == 3


def test_wrist_columns_are_identically_zero() -> None:
    q = np.random.default_rng(2).uniform(-1.0, 1.0, N_DOF)
    jac = analytical_constraint_jacobian(q, _params())
    for col in WRIST_COLUMNS:
        assert np.max(np.abs(jac[:, col])) == 0.0


def test_every_closed_solution_has_club_angle_plus_minus_half_pi() -> None:
    for sol in _closed_solutions():
        wrapped = (sol[7] + np.pi) % (2.0 * np.pi) - np.pi
        assert abs(abs(wrapped) - np.pi / 2.0) < 1e-6


@pytest.mark.xfail(
    strict=True,
    reason="forward_kinematics flips the y sign of club points (NSX-12 finding)",
)
def test_forward_kinematics_club_points_follow_the_club_direction() -> None:
    p = _params()
    q = np.zeros(N_DOF)
    q[7] = 0.4  # cos != 0
    fk = forward_kinematics(q, p)
    direction = np.array([np.sin(q[7]), -np.cos(q[7])])
    got = np.array(fk["grip_left"]) - np.array(fk["grip_right"])
    assert np.allclose(got, (p.grip_left - p.grip_right) * direction)
