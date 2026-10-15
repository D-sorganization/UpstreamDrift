"""Rank, sign and closure characterisation of ``golfer_constraints`` (#12608).

NSX-12 (#12557) found that ``forward_kinematics`` flipped the y component of
every club point, pinning every closed solution to a club angle of ``+-pi/2``.
These tests pin the corrected behaviour against sources that are independent
of the code under test:

* the NSX-00 planar two-arm model (``loop_nullspace.planar_model``), whose
  closure ``Phi = H_trail - G`` is the golfer's rows 0-1 once the golfer's
  right arm is mapped to the planar lead arm (the club is anchored there);
* the analytic shaft direction ``u(theta) = (sin theta, -cos theta)``;
* central finite differences written out here.

Closure decision (documented in ``golfer_kinematics``): the hands do NOT enter
the closure.  ``rh``/``lh`` are forearm end points, the club is pinned at both
wrists (closure kind ``PLANAR_POINT2``, as in NSX-00) and the wrist angles are
free hand DOFs, so their Jacobian columns are identically zero.  Rows 2-3 are
an orthogonal rotation of rows 0-1, so the Jacobian has rank 2 on the
constraint manifold (rank 3 off it, where the rotation's derivative adds a
term proportional to ``Phi`` in the club-angle column).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    planar_dynamics,
)
from src.shared.python.pendulum_simulator.golfer_constraints import (
    analytical_constraint_jacobian,
    constraint_vector,
    numerical_constraint_jacobian,
)
from src.shared.python.pendulum_simulator.golfer_dynamics import (
    analytical_fk_jacobians,
)
from src.shared.python.pendulum_simulator.golfer_kinematics import forward_kinematics
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams

pytestmark = pytest.mark.unit

WRIST_COLUMNS = (3, 6)
CLUB = 7
FD_STEP = 1e-6
FD_TOL = 1e-8
RANK_TOL = 1e-8


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


def _planar_params(p: GolferParams) -> PlanarGolferParams:
    """Planar model with the golfer's geometry (masses are irrelevant here)."""
    return PlanarGolferParams(
        shoulder_separation=p.d_rs + p.d_ls,
        upper_arm_length=p.L_r_upper,
        forearm_length=p.L_r_fore,
        upper_arm_mass=1.0,
        forearm_mass=1.0,
        club_mass=1.0,
        club_length=p.L_club,
        club_com_from_lead_hand=0.5 * p.L_club,
        trail_grip_from_lead_hand=p.grip_left - p.grip_right,
        gravity=9.81,
    )


# Golfer q = A @ q_planar + b with theta_hub = pi, so the right shoulder sits
# at x = -d (planar lead) and the left at x = +d (planar trail).  Planar angles
# are relative along each chain; golfer arm angles are relative to the hub and
# the golfer club angle is absolute.
_A = np.zeros((N_DOF, 5))
_A[1, 0] = _A[2, 1] = 1.0  # lead shoulder, elbow -> right arm
_A[4, 3] = _A[5, 4] = 1.0  # trail shoulder, elbow -> left arm
_A[CLUB, 0:3] = 1.0  # absolute club angle = q1 + q2 + q3
_B = np.zeros(N_DOF)
_B[0] = np.pi
_B[1] = _B[4] = -np.pi


def _golfer_q(q_planar: np.ndarray, wrists: tuple[float, float]) -> np.ndarray:
    q = _A @ q_planar + _B
    q[WRIST_COLUMNS[0]], q[WRIST_COLUMNS[1]] = wrists
    return q


def _central_fd(func, q: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    cols = []
    for j in range(q.shape[0]):
        step = np.zeros_like(q)
        step[j] = FD_STEP
        cols.append((np.asarray(func(q + step)) - np.asarray(func(q - step))) / 2e-6)
    return np.stack(cols, axis=1)


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


def test_analytic_jacobian_matches_central_difference_on_and_off_manifold() -> None:
    p = _params()
    rng = np.random.default_rng(3)
    for q in [rng.uniform(-1.0, 1.0, N_DOF), *_closed_solutions(4)]:
        fd = _central_fd(lambda x: constraint_vector(x, p), q)
        assert np.max(np.abs(analytical_constraint_jacobian(q, p) - fd)) < FD_TOL


def test_club_point_jacobians_match_central_difference() -> None:
    p = _params()
    q = np.random.default_rng(4).uniform(-1.0, 1.0, N_DOF)
    jac = analytical_fk_jacobians(q, p)

    def tip(x: np.ndarray) -> np.ndarray:
        return np.array(forward_kinematics(x, p)["club_tip"])

    def com(x: np.ndarray) -> np.ndarray:
        fk = forward_kinematics(x, p)
        return 0.5 * (np.array(fk["club_base"]) + np.array(fk["club_tip"]))

    assert np.max(np.abs(jac["club_tip"] - _central_fd(tip, q))) < FD_TOL
    assert np.max(np.abs(jac["club_com"] - _central_fd(com, q))) < FD_TOL


def test_rank_is_two_on_the_constraint_manifold() -> None:
    for sol in _closed_solutions():
        jac = analytical_constraint_jacobian(sol, _params())
        assert np.linalg.matrix_rank(jac, tol=RANK_TOL) == 2


def test_rank_is_three_off_the_manifold() -> None:
    q = np.random.default_rng(1).uniform(-1.0, 1.0, N_DOF)
    jac = analytical_constraint_jacobian(q, _params())
    assert jac.shape == (4, N_DOF)
    assert np.linalg.matrix_rank(jac, tol=RANK_TOL) == 3


def test_rows_two_three_rotate_rows_zero_one() -> None:
    p, q = _params(), np.random.default_rng(5).uniform(-2.0, 2.0, N_DOF)
    phi = constraint_vector(q, p)
    u = np.array([np.sin(q[CLUB]), -np.cos(q[CLUB])])
    rot = np.array([[-u[1], u[0]], u])  # rows: club_perp, club_dir
    assert np.allclose(phi[2:], rot @ phi[:2], atol=1e-14)


def test_wrist_columns_are_identically_zero() -> None:
    q = np.random.default_rng(2).uniform(-1.0, 1.0, N_DOF)
    jac = analytical_constraint_jacobian(q, _params())
    for col in WRIST_COLUMNS:
        assert np.max(np.abs(jac[:, col])) == 0.0


def test_closed_solutions_are_not_pinned_to_half_pi() -> None:
    cosines = [abs(np.cos(sol[CLUB])) for sol in _closed_solutions()]
    assert max(cosines) > 0.5


def test_forward_kinematics_club_points_follow_the_club_direction() -> None:
    p = _params()
    q = np.zeros(N_DOF)
    q[CLUB] = 0.4  # cos != 0
    fk = forward_kinematics(q, p)
    direction = np.array([np.sin(q[CLUB]), -np.cos(q[CLUB])])
    rh = np.array(fk["rh"])
    assert np.allclose(
        np.array(fk["grip_left"]) - rh, (p.grip_left - p.grip_right) * direction
    )
    assert np.allclose(np.array(fk["club_base"]) - rh, -p.grip_right * direction)
    assert np.allclose(
        np.array(fk["club_tip"]) - rh, (p.L_club - p.grip_right) * direction
    )


def test_closure_matches_planar_model_closure_and_jacobian() -> None:
    p = _params()
    pp = _planar_params(p)
    rng = np.random.default_rng(6)
    for _ in range(20):
        qp = rng.uniform(-np.pi, np.pi, 5)
        q = _golfer_q(qp, tuple(rng.uniform(-1.0, 1.0, 2)))
        planar = planar_dynamics(qp, np.zeros(5), pp)
        assert np.allclose(constraint_vector(q, p)[:2], planar["phi"], atol=1e-12)
        jac = analytical_constraint_jacobian(q, p)
        assert np.allclose(jac[:2] @ _A, planar["Jc"], atol=1e-12)


def test_planar_closed_configuration_closes_the_golfer() -> None:
    p = _params()
    pp = _planar_params(p)
    res = least_squares(
        lambda x: planar_dynamics(
            np.array([0.1, 1.4, -0.3, x[0], x[1]]), np.zeros(5), pp
        )["phi"],
        np.array([0.2, 1.0]),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )
    qp = np.array([0.1, 1.4, -0.3, *res.x])
    assert np.max(np.abs(planar_dynamics(qp, np.zeros(5), pp)["phi"])) < 1e-12
    q = _golfer_q(qp, (0.3, -0.2))
    assert abs(np.cos(q[CLUB])) > 0.1
    assert np.max(np.abs(constraint_vector(q, p))) < 1e-12
