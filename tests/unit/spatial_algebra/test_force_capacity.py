"""Tests for zero-velocity dynamically consistent force capacity (#14047, CAP-7)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.spatial_algebra.operational_space import (
    acceleration_capacity,
    force_capacity,
    mobility_matrix,
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


def test_two_link_force_capacity_matches_khatib_duality():
    J, M, adj, det = _arm()
    cap_f = force_capacity(J, M, np.eye(2), LO, HI)
    cap_a = acceleration_capacity(J, M, np.eye(2), LO, HI)

    # Upper 2x2 task space is fully unconstrained and invertible
    J2 = J[:2]
    Minv = adj / det
    W2 = J2 @ Minv @ J2.T
    W2_inv = np.linalg.inv(W2)

    T = np.diag((HI - LO) / 2)
    mid = (HI + LO) / 2
    Ga2 = J2 @ Minv @ T
    Qa2 = Ga2 @ Ga2.T
    Qf_expected = W2_inv @ Qa2 @ W2_inv

    # Reconstructed force capacity ellipsoid matrix
    recon = (cap_f.axes * cap_f.radii**2) @ cap_f.axes.T
    np.testing.assert_allclose(recon[:2, :2], Qf_expected, atol=1e-9)
    # The planar arm has no z authority: exactly one flagged axis
    assert int(cap_f.singular.sum()) == 1
    # Check centre shift
    centre_expected = W2_inv @ (J2 @ Minv @ mid)
    np.testing.assert_allclose(cap_f.centre[:2], centre_expected, atol=1e-9)
    np.testing.assert_allclose(cap_f.centre[2], 0.0, atol=1e-12)


def test_one_dof_force_radius_independent_of_mass():
    J = np.array([[2.0], [0.0], [0.0]])
    lo, hi = np.array([-6.0]), np.array([6.0])
    B = np.eye(1)

    cap_m4 = force_capacity(J, np.array([[4.0]]), B, lo, hi)
    cap_m10 = force_capacity(J, np.array([[10.0]]), B, lo, hi)

    # In 1-DoF, Khatib duality cancels mass: W^-1 * G_a = (M/J^2) * (J * tau / M) = tau / J
    # 6.0 / 2.0 = 3.0 N
    assert cap_m4.radii.max() == pytest.approx(3.0)
    assert cap_m10.radii.max() == pytest.approx(3.0)
    np.testing.assert_allclose(cap_m4.centre, 0.0, atol=1e-15)
    np.testing.assert_allclose(cap_m10.centre, 0.0, atol=1e-15)


def test_symmetric_limits_zero_centre_asymmetric_shift():
    J, M, _, _ = _arm()
    sym = force_capacity(J, M, np.eye(2), -HI, HI)
    np.testing.assert_allclose(sym.centre, 0.0, atol=1e-12)
    asym = force_capacity(J, M, np.eye(2), LO, HI)
    assert np.linalg.norm(asym.centre) > 0


def test_closure_keeps_force_capacity_consistent():
    rng = np.random.default_rng(7)
    nv = 4
    Lm = rng.normal(size=(nv, nv))
    M = Lm @ Lm.T + nv * np.eye(nv)
    J = rng.normal(size=(3, nv))
    Jc = rng.normal(size=(1, nv))
    B = np.eye(nv)
    lo, hi = -np.full(nv, 10.0), np.full(nv, 10.0)

    cap_constrained = force_capacity(J, M, B, lo, hi, Jc=Jc)
    cap_free = force_capacity(J, M, B, lo, hi)

    # Closure removes degrees of freedom
    assert np.all(cap_constrained.radii <= cap_free.radii + 1e-9)
    assert cap_constrained.radii.max() < cap_free.radii.max()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lo": [1.0, -1.0], "hi": [1.0, 1.0]},  # lo not below zero
        {"lo": [-1.0, -1.0], "hi": [1.0, -0.5]},  # hi not above zero
        {"lo": [-1.0], "hi": [1.0]},  # wrong length
        {"lo": [-1.0, -np.inf], "hi": [1.0, 1.0]},  # non-finite
        {"lo": [-1.0, -1.0], "hi": [1.0, 1.0], "B": np.eye(3)},  # B shape
    ],
)
def test_preconditions_raise(kwargs: dict):
    J, M, _, _ = _arm()
    B = kwargs.get("B", np.eye(2))
    with pytest.raises(ValueError):
        force_capacity(J, M, B, kwargs["lo"], kwargs["hi"])
