"""Shared convex-QP front end over the conic core (NSX-24, #12567)."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from src.shared.python.loop_nullspace.qp_core import (
    QPConstraints,
    friction_cone_rows,
    solve_qp,
)
from src.shared.python.loop_nullspace.solvers import ConicSettings, SolveStatus

TIGHT = ConicSettings(tol=1e-11)

pytestmark = pytest.mark.unit


def test_equality_qp_matches_closed_form_min_norm() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 7))
    b = rng.normal(size=3)
    h = rng.uniform(0.5, 4.0, size=7)
    res = solve_qp(
        np.diag(h), np.zeros(7), QPConstraints(A_eq=a, b_eq=b), settings=TIGHT
    )
    assert res.status is SolveStatus.OPTIMAL
    hinv = 1.0 / h
    expected = hinv * (a.T @ np.linalg.solve((a * hinv) @ a.T, b))
    np.testing.assert_allclose(res.x, expected, atol=1e-8)


def test_fixed_bounds_become_equalities_and_hold_exactly() -> None:
    res = solve_qp(
        np.eye(3),
        np.zeros(3),
        QPConstraints(
            A_eq=np.ones((1, 3)),
            b_eq=np.array([3.0]),
            lo=np.array([0.0, -np.inf, 1.0]),
            hi=np.array([0.0, np.inf, 1.0]),
        ),
    )
    assert res.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(res.x, [0.0, 2.0, 1.0], atol=1e-7)


def test_inequality_and_box_active() -> None:
    res = solve_qp(
        np.eye(2),
        np.array([-4.0, -4.0]),
        QPConstraints(
            A_ub=np.array([[1.0, 1.0]]),
            b_ub=np.array([2.0]),
            lo=np.array([0.0, 0.0]),
            hi=np.array([5.0, 5.0]),
        ),
        settings=TIGHT,
    )
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-7)


def test_infeasible_is_reported_not_repaired() -> None:
    res = solve_qp(
        np.eye(1),
        np.zeros(1),
        QPConstraints(
            A_eq=np.ones((1, 1)),
            b_eq=np.array([5.0]),
            lo=np.array([0.0]),
            hi=np.array([1.0]),
        ),
    )
    assert res.status is SolveStatus.INFEASIBLE
    assert res.x is None


def test_exact_coulomb_cone_binds_at_mu() -> None:
    # Force wants a tangential 2 with normal 1; mu = 0.5 caps the tangent at 0.5*fz.
    block, a_uni, b_uni = friction_cone_rows(3, (0, 1, 2), np.eye(3), 0.5)
    res = solve_qp(
        np.eye(3),
        np.array([-2.0, 0.0, -1.0]),
        QPConstraints(A_ub=a_uni, b_ub=b_uni, cones=(block,)),
        settings=TIGHT,
    )
    fx, fy, fz = res.x
    assert fz >= -1e-8
    assert np.hypot(fx, fy) <= 0.5 * fz + 1e-7
    assert np.hypot(fx, fy) == pytest.approx(0.5 * fz, abs=1e-6)


def test_frictionless_cone_still_enforces_unilateral_normal() -> None:
    block, a_uni, b_uni = friction_cone_rows(3, (0, 1, 2), np.eye(3), 0.0)
    res = solve_qp(
        np.eye(3),
        np.array([0.0, 0.0, 3.0]),
        QPConstraints(A_ub=a_uni, b_ub=b_uni, cones=(block,)),
    )
    assert res.x[2] >= -1e-8


def test_non_symmetric_hessian_rejected() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        solve_qp(np.array([[1.0, 1.0], [0.0, 1.0]]), np.zeros(2))


def test_missing_backend_raises_actionable_importerror_no_fallback(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "clarabel" or name.startswith("clarabel."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="loop-nullspace"):
        solve_qp(np.eye(1), np.zeros(1))
