"""NSX-30: ``constrained_acceleration`` is the single KKT-projector solve (#12570).

The reference for every check is an independent augmented KKT block solve
(``[[M, -J^T], [J, 0]] [a; lam] = [load; -Jdot v]``), never the projector under
test.  Matrices come from the NSX-00 planar goldens.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.constrained_dynamics import (
    Baumgarte,
    closure_acceleration_residual,
    constrained_acceleration,
    kkt_projector,
)
from src.shared.python.loop_nullspace.fixtures import (
    planar_golfer_snapshot,
    straight_arm_snapshot,
)

pytestmark = pytest.mark.unit

TOL = 1e-9


def _scaled(x: np.ndarray, scale: float) -> float:
    return float(np.max(np.abs(x))) / max(1.0, scale)


def _augmented_reference(M, load, J, Jdot_v):
    """Independent full-rank KKT block solve for ``M a = load + J^T lam``."""
    n, m = M.shape[0], J.shape[0]
    kkt = np.block([[M, -J.T], [J, np.zeros((m, m))]])
    sol = np.linalg.solve(kkt, np.concatenate([load, -Jdot_v]))
    return sol[:n], sol[n:]


@pytest.fixture(scope="module")
def generic():
    return planar_golfer_snapshot().snapshot


def test_matches_independent_kkt_solve_on_the_planar_golden(generic) -> None:
    s = generic
    load = s.S.T @ np.array([1.3, -0.7, 0.4, 2.1, -1.9]) - s.h
    a, lam = constrained_acceleration(
        s.M, s.h, s.S.T @ np.array([1.3, -0.7, 0.4, 2.1, -1.9]), s.J_c, s.Jdot_v
    )
    a_ref, lam_ref = _augmented_reference(s.M, load, s.J_c, s.Jdot_v)
    assert _scaled(a - a_ref, np.max(np.abs(a_ref))) < TOL
    assert _scaled(lam - lam_ref, np.max(np.abs(lam_ref))) < TOL


def test_reproduces_the_golden_acceleration_for_the_golden_torque() -> None:
    case = planar_golfer_snapshot()
    s, g = case.snapshot, case.golden
    tau = g.array("min_metric_torque", "minv", "tau_actuated")
    a, _ = constrained_acceleration(s.M, s.h, s.S.T @ tau, s.J_c, s.Jdot_v)
    assert _scaled(a - s.a, np.max(np.abs(s.a))) < 1e-8


def test_closure_postcondition_holds(generic) -> None:
    s = generic
    tau = s.S.T @ np.arange(1.0, 6.0)
    a, _ = constrained_acceleration(s.M, s.h, tau, s.J_c, s.Jdot_v)
    res = closure_acceleration_residual(s.J_c, a, s.Jdot_v)
    assert _scaled(res, np.max(np.abs(s.Jdot_v))) < TOL


def test_reaction_satisfies_the_equation_of_motion(generic) -> None:
    s = generic
    tau = s.S.T @ np.arange(1.0, 6.0)
    a, lam = constrained_acceleration(s.M, s.h, tau, s.J_c, s.Jdot_v)
    resid = s.M @ a + s.h - tau - s.J_c.T @ lam
    assert _scaled(resid, np.max(np.abs(s.h))) < TOL


def test_projector_formula_and_idempotent_structure(generic) -> None:
    s = generic
    p = kkt_projector(s.M, s.J_c)
    minv = np.linalg.inv(s.M)
    lam_c = np.linalg.inv(s.J_c @ minv @ s.J_c.T)
    expected = minv @ (np.eye(5) - s.J_c.T @ lam_c @ s.J_c @ minv)
    assert _scaled(p - expected, np.max(np.abs(expected))) < TOL
    assert np.max(np.abs(s.J_c @ p)) < 1e-9
    assert np.max(np.abs(p - p.T)) < 1e-9


def test_rank_deficient_acceleration_is_unique_and_closed() -> None:
    s = straight_arm_snapshot().snapshot
    tau = s.S.T @ np.array([0.5, -1.0, 0.25, 1.5, 0.75])
    a, lam = constrained_acceleration(s.M, s.h, tau, s.J_c, s.Jdot_v)
    res = closure_acceleration_residual(s.J_c, a, s.Jdot_v)
    assert _scaled(res, max(1.0, np.max(np.abs(s.Jdot_v)))) < TOL
    gauge = np.linalg.svd(s.J_c.T)[2][-1]  # a lambda direction with J^T w = 0
    lam2 = lam + 3.0 * gauge
    balance = s.M @ a + s.h - tau - s.J_c.T @ lam2
    assert _scaled(balance, np.max(np.abs(s.h))) < 1e-8


def test_zero_closure_rows_reduce_to_unconstrained_dynamics() -> None:
    m = np.diag([2.0, 3.0])
    a, lam = constrained_acceleration(
        m, np.array([1.0, 1.0]), np.array([3.0, 7.0]), np.zeros((0, 2)), np.zeros(0)
    )
    np.testing.assert_allclose(a, [1.0, 2.0], atol=1e-14)
    assert lam.shape == (0,)


def test_baumgarte_terms_are_declared_and_change_only_the_correction(generic) -> None:
    s = generic
    tau = s.S.T @ np.ones(5)
    plain, _ = constrained_acceleration(s.M, s.h, tau, s.J_c, s.Jdot_v)
    zero = Baumgarte(
        alpha=5.0,
        beta=5.0,
        position_error=np.zeros(2),
        velocity_error=np.zeros(2),
    )
    same, _ = constrained_acceleration(
        s.M, s.h, tau, s.J_c, s.Jdot_v, baumgarte=zero
    )
    np.testing.assert_allclose(same, plain, atol=1e-12)
    phi = np.array([1e-3, -2e-3])
    gains = Baumgarte(
        alpha=5.0, beta=4.0, position_error=phi, velocity_error=np.zeros(2)
    )
    a, _ = constrained_acceleration(s.M, s.h, tau, s.J_c, s.Jdot_v, baumgarte=gains)
    expected = -(s.Jdot_v + 16.0 * phi)
    assert np.max(np.abs(s.J_c @ a - expected)) < 1e-9


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"M": np.eye(3)}, "shape"),
        ({"h": np.zeros(4)}, "h must"),
        ({"tau": np.zeros(4)}, "tau must"),
        ({"J_c": np.zeros((2, 4))}, "J_c must"),
        ({"Jdot_v": np.zeros(3)}, "Jdot_v must"),
        ({"M": -np.eye(5)}, "positive definite"),
        ({"h": np.full(5, np.nan)}, "finite"),
    ],
)
def test_preconditions_raise_value_error(generic, kwargs, match) -> None:
    s = generic
    args = {
        "M": s.M,
        "h": s.h,
        "tau": np.zeros(5),
        "J_c": s.J_c,
        "Jdot_v": s.Jdot_v,
    } | kwargs
    with pytest.raises(ValueError, match=match):
        constrained_acceleration(**args)


def test_baumgarte_rejects_negative_gains() -> None:
    with pytest.raises(ValueError, match="alpha"):
        Baumgarte(alpha=-1.0, beta=1.0, position_error=np.zeros(2), velocity_error=np.zeros(2))
