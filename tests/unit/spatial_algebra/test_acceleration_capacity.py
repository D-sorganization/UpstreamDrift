"""Tests for acceleration capacity and drift acceleration (#14040, CAP-1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.constrained_dynamics import (
    closure_acceleration_residual,
    constrained_acceleration,
)
from src.shared.python.spatial_algebra.operational_space import (
    acceleration_capacity,
    drift_acceleration,
)

pytestmark = pytest.mark.unit

Q1, Q2 = 0.4, 0.9
LO = np.array([-30.0, -12.0])
HI = np.array([50.0, 20.0])


def _arm():
    """Planar 2-link arm, unit point masses at link ends, unit lengths.

    Returns (J3, M, adj_M, det_M) with the tip Jacobian padded by a zero z row.
    The mass matrix is written out by hand; its inverse via the 2x2 adjugate.
    """
    c1, s1 = np.cos(Q1), np.sin(Q1)
    c12, s12 = np.cos(Q1 + Q2), np.sin(Q1 + Q2)
    c2 = np.cos(Q2)
    M = np.array([[3 + 2 * c2, 1 + c2], [1 + c2, 1.0]])
    J = np.array([[-s1 - s12, -s12], [c1 + c12, c12], [0.0, 0.0]])
    det = M[0, 0] * M[1, 1] - M[0, 1] ** 2
    adj = np.array([[M[1, 1], -M[0, 1]], [-M[0, 1], M[0, 0]]])
    return J, M, adj, det


def test_two_link_matches_hand_derivation():
    J, M, adj, det = _arm()
    cap = acceleration_capacity(J, M, np.eye(2), LO, HI)
    T = np.diag((HI - LO) / 2)
    G = J @ (adj / det) @ T
    q_expected = G @ G.T
    recon = (cap.axes * cap.radii**2) @ cap.axes.T
    np.testing.assert_allclose(recon, q_expected, atol=1e-9)
    # the planar arm has no z authority: exactly one flagged axis
    assert int(cap.singular.sum()) == 1
    mid = (HI + LO) / 2
    np.testing.assert_allclose(cap.centre, J @ (adj / det) @ mid, atol=1e-12)


def test_one_dof_radius_is_j_t_over_m():
    J = np.array([[2.0], [0.0], [0.0]])
    cap = acceleration_capacity(J, np.array([[4.0]]), np.eye(1), [-5.0], [5.0])
    assert cap.radii.max() == pytest.approx(2.0 * 5.0 / 4.0)
    np.testing.assert_allclose(cap.centre, 0.0, atol=1e-15)


def test_symmetric_limits_zero_centre_asymmetric_shift():
    J, M, _, _ = _arm()
    sym = acceleration_capacity(J, M, np.eye(2), -HI, HI)
    np.testing.assert_allclose(sym.centre, 0.0, atol=1e-12)
    asym = acceleration_capacity(J, M, np.eye(2), LO, HI)
    expect = J @ np.linalg.solve(M, (HI + LO) / 2)
    np.testing.assert_allclose(asym.centre, expect, atol=1e-12)
    assert np.linalg.norm(asym.centre) > 0


def test_closure_keeps_capacity_consistent():
    rng = np.random.default_rng(3)
    nv = 4
    Lm = rng.normal(size=(nv, nv))
    M = Lm @ Lm.T + nv * np.eye(nv)
    J = rng.normal(size=(3, nv))
    Jc = rng.normal(size=(1, nv))
    B = np.eye(nv)
    lo, hi = -np.full(nv, 10.0), np.full(nv, 10.0)
    cap = acceleration_capacity(J, M, B, lo, hi, Jc=Jc)
    free = acceleration_capacity(J, M, B, lo, hi)
    # closure only removes authority
    assert np.all(cap.radii <= free.radii + 1e-9) and cap.radii.max() < (
        free.radii.max()
    )
    # B*T part of the acceleration satisfies Jc a = 0
    from src.shared.python.loop_nullspace.constrained_dynamics import kkt_projector

    a = kkt_projector(M, Jc) @ (B @ np.diag(hi) @ rng.uniform(-1, 1, nv))
    np.testing.assert_allclose(Jc @ a, 0.0, atol=1e-9)


def test_ellipsoid_inside_zonotope_and_zonotope_inside_box():
    J, M, _, _ = _arm()
    J = J[:2]  # 2 x 2, full authority
    rng = np.random.default_rng(0)
    J3 = np.vstack([J, np.zeros((1, 2))])
    cap = acceleration_capacity(J3, M, np.eye(2), LO, HI)
    A = np.linalg.inv(M)
    T = (HI - LO) / 2
    mid = (HI + LO) / 2
    G = J @ A @ np.diag(T)
    # (a) sampled box torques map inside the circumscribing box
    taus = rng.uniform(LO, HI, size=(2000, 2))
    pts = taus @ (J @ A).T
    half = np.abs(G).sum(axis=1)
    centre = J @ A @ mid
    assert np.all(np.abs(pts - centre) <= half + 1e-9)
    # (b) every boundary point of the ellipsoid is attainable by a box torque
    theta = rng.uniform(0, 2 * np.pi, 2000)
    u = np.stack([np.cos(theta), np.sin(theta)])
    x = G @ u  # points on the ellipsoid boundary (c + x)
    u_back = np.linalg.solve(G, x)
    tau = mid[:, None] + T[:, None] * u_back
    assert np.all(tau >= LO[:, None] - 1e-9) and np.all(tau <= HI[:, None] + 1e-9)
    # and the reported ellipsoid describes the same set
    r = cap.axes[:2, :2].T @ x
    s = (r / np.maximum(cap.radii[cap.radii > 1e-6][:, None], 1e-12)) ** 2
    assert s.sum(axis=0).max() <= 1 + 1e-9


def test_drift_matches_constrained_solve_with_zero_tau():
    rng = np.random.default_rng(5)
    nv = 5
    Lm = rng.normal(size=(nv, nv))
    M = Lm @ Lm.T + nv * np.eye(nv)
    h = rng.normal(size=nv)
    J = rng.normal(size=(3, nv))
    Jdot_v = rng.normal(size=3)
    Jc = rng.normal(size=(2, nv))
    Jcdot_v = rng.normal(size=2)
    a, _ = constrained_acceleration(M, h, np.zeros(nv), Jc, Jcdot_v)
    np.testing.assert_allclose(
        closure_acceleration_residual(Jc, a, Jcdot_v), 0, atol=1e-9
    )
    c = drift_acceleration(J, M, h, Jdot_v, Jc=Jc, Jcdot_v=Jcdot_v)
    np.testing.assert_allclose(c, J @ a + Jdot_v, atol=1e-9)
    # without closure: a0 = -M^-1 h
    c0 = drift_acceleration(J, M, h, Jdot_v)
    np.testing.assert_allclose(c0, -J @ np.linalg.solve(M, h) + Jdot_v, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lo": [1.0, -1.0], "hi": [1.0, 1.0]},  # lo not below zero
        {"lo": [-1.0, -1.0], "hi": [1.0, -0.5]},  # hi not above zero
        {"lo": [-1.0], "hi": [1.0]},  # wrong length
        {"lo": [-1.0, -np.inf], "hi": [1.0, 1.0]},  # non-finite
        {"lo": [-1.0, -1.0], "hi": [1.0, 1.0], "B": np.eye(3)},  # B shape
        {"lo": [-1.0, -1.0], "hi": [1.0, 1.0], "M": -np.eye(2)},  # not SPD
    ],
)
def test_invalid_inputs_raise(kwargs):
    J, M, _, _ = _arm()
    kw = dict(kwargs)
    with pytest.raises(ValueError):
        acceleration_capacity(
            J, kw.get("M", M), kw.get("B", np.eye(2)), kw["lo"], kw["hi"]
        )


def test_drift_invalid_inputs_raise():
    J, M, _, _ = _arm()
    with pytest.raises(ValueError):
        drift_acceleration(J, M, np.zeros(3), np.zeros(3))  # h wrong length
    with pytest.raises(ValueError):
        drift_acceleration(J, M, np.zeros(2), np.zeros(2))  # Jdot_v wrong length
