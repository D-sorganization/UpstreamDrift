"""Tests for operational-space mobility matrices and ellipsoids (#14002)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.spatial_algebra.operational_space import (
    Ellipsoid,
    force_ellipsoid,
    mobility_matrix,
    transport_spatial_mobility,
    velocity_ellipsoid,
)
from src.shared.python.loop_nullspace.decomposition import CharacteristicScales
from src.shared.python.spatial_algebra.spatial_vectors import skew

pytestmark = pytest.mark.unit


def _arm(q1: float, q2: float):
    """Planar 2-link arm: point masses m1, m2 at link ends, lengths l1, l2."""
    m1, m2, l1, l2 = 1.5, 0.8, 0.6, 0.4
    c2 = np.cos(q2)
    a = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
    b = m2 * l2**2 + m2 * l1 * l2 * c2
    d = m2 * l2**2
    M = np.array([[a, b], [b, d]])
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    J = np.array(
        [
            [-l1 * s1 - l2 * s12, -l2 * s12],
            [l1 * c1 + l2 * c12, l2 * c12],
        ]
    )
    return J, M


def _two_masses():
    # nv = 6: mass A (dofs 0-2), mass B (dofs 3-5)
    M = np.diag([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    J = np.hstack([np.eye(3), np.zeros((3, 3))])
    Jc = np.hstack([np.eye(3), -np.eye(3)])
    return J, M, Jc


def test_point_mass_is_identity_over_mass():
    m = 2.5
    result = mobility_matrix(np.eye(3), m * np.eye(3))
    np.testing.assert_allclose(result, np.eye(3) / m, atol=1e-14)


def test_planar_arm_matches_closed_form():
    J, M = _arm(0.3, 0.9)
    expected = J @ np.linalg.inv(M) @ J.T
    J3 = np.vstack([J, np.zeros((1, 2))])
    result = mobility_matrix(J3, M)
    np.testing.assert_allclose(result[:2, :2], expected, rtol=1e-12)
    assert result[2, 2] == 0.0


def test_constrained_is_psd_and_loewner_below_unconstrained():
    rng = np.random.default_rng(1)
    nv = 7
    B = rng.normal(size=(nv, nv))
    M = B @ B.T + nv * np.eye(nv)
    J = rng.normal(size=(3, nv))
    Jc = rng.normal(size=(2, nv))
    u = mobility_matrix(J, M)
    c = mobility_matrix(J, M, Jc)
    assert np.linalg.eigvalsh(c).min() >= -1e-12
    assert np.linalg.eigvalsh(u - c).min() >= -1e-12
    np.testing.assert_allclose(c, c.T, atol=1e-15)


def test_rigidly_locked_direction_has_zero_mobility():
    J, M, Jc = _two_masses()
    result = mobility_matrix(J, M, Jc)
    # Joined masses move together: 1/(m_A + m_B) = 0.2, not 1/2.
    np.testing.assert_allclose(result, np.eye(3) / 5.0, atol=1e-12)
    # Fully lock mass A to ground as well: nothing can move.
    Jc_all = np.vstack([Jc, np.hstack([np.eye(3), np.zeros((3, 3))])])
    np.testing.assert_allclose(mobility_matrix(J, M, Jc_all), 0.0, atol=1e-12)


def test_rank_deficient_constraint_is_handled():
    J, M, Jc = _two_masses()
    Jc_dup = np.vstack([Jc, Jc])
    np.testing.assert_allclose(
        mobility_matrix(J, M, Jc_dup), mobility_matrix(J, M, Jc), atol=1e-10
    )


def test_ellipsoids_orthonormal_and_reciprocal():
    J, M = _arm(0.4, 1.1)
    mob = mobility_matrix(np.vstack([J, [[1.0, 0.5]]]), M)
    vel = velocity_ellipsoid(mob)
    frc = force_ellipsoid(mob)
    assert isinstance(vel, Ellipsoid)
    np.testing.assert_allclose(vel.axes.T @ vel.axes, np.eye(3), atol=1e-12)
    assert np.all(vel.radii >= 0)
    ok = ~frc.singular
    np.testing.assert_allclose(frc.radii[ok] * vel.radii[ok], 1.0, rtol=1e-9)


def test_force_ellipsoid_flags_singular_direction():
    J, M, Jc = _two_masses()
    Jc_all = np.vstack([Jc, np.hstack([np.eye(3), np.zeros((3, 3))])])
    frc = force_ellipsoid(mobility_matrix(J, M, Jc_all))
    assert frc.singular.all()
    assert np.all(np.isfinite(frc.radii))


def test_characteristic_length_scales_rotational_block():
    rng = np.random.default_rng(2)
    J = rng.normal(size=(6, 6))
    M = np.eye(6) * 2.0
    base = mobility_matrix(J, M)
    scaled = mobility_matrix(J, M, scales=CharacteristicScales())
    s = np.diag([1, 1, 1, 0.05, 0.05, 0.05])
    # angular rows multiplied by L = 5/100 = 0.05: S base S
    np.testing.assert_allclose(scaled, s @ base @ s, rtol=1e-12)


def test_point_mass_plus_rotor_has_uniform_scaled_mobility():
    """Point mass m plus rotor of inertia I = m*L^2 has uniform 1/m mobility across all 6x6 blocks (#14053)."""
    m = 2.0
    scales = CharacteristicScales(force=100.0, moment=5.0)
    length = scales.moment / scales.force  # 0.05 m
    inertia_rotor = m * length**2  # 0.005 kg*m^2
    M = np.diag([m, m, m, inertia_rotor, inertia_rotor, inertia_rotor])
    J = np.eye(6)
    scaled = mobility_matrix(J, M, scales=scales)
    # Scaled 6x6 mobility should have uniform kg^-1 units equal to (1/m) * I_6
    expected = (1.0 / m) * np.eye(6)
    np.testing.assert_allclose(scaled, expected, atol=1e-12)


@pytest.mark.parametrize(
    "J,M,Jc,kwargs",
    [
        (np.ones((4, 4)), np.eye(4), None, {}),  # k not 3 or 6
        (np.ones((3, 4)), np.eye(5), None, {}),  # shape mismatch
        (np.ones((3, 4)), np.ones((4, 4)) - 5 * np.eye(4), None, {}),  # not SPD
        (np.ones((3, 4)), np.array([[1, 2, 0, 0]] * 4, float), None, {}),  # asym
        (np.full((3, 4), np.nan), np.eye(4), None, {}),
        (np.ones((3, 4)), np.eye(4), np.ones((2, 5)), {}),  # Jc width
        (np.ones((3, 4)), np.eye(4), None, {"scales": CharacteristicScales()}),  # k=3
    ],
)
def test_bad_inputs_raise_value_error(J, M, Jc, kwargs):
    with pytest.raises(ValueError):
        mobility_matrix(J, M, Jc, **kwargs)


def test_transport_matches_rigid_weld_constrained_mobility():
    rng = np.random.default_rng(3)
    nv = 8
    B = rng.normal(size=(nv, nv))
    M = B @ B.T + nv * np.eye(nv)
    Ja = rng.normal(size=(6, nv))
    Jc = rng.normal(size=(3, nv))
    r = np.array([0.1, -0.2, 0.05])
    X = np.eye(6)
    X[:3, 3:] = -skew(r)
    Jb = X @ Ja  # point b rigidly welded to a
    mob_a = mobility_matrix(Ja, M, Jc)
    mob_b = mobility_matrix(Jb, M, Jc)
    np.testing.assert_allclose(mob_b, X @ mob_a @ X.T, atol=1e-10)
    np.testing.assert_allclose(transport_spatial_mobility(mob_a, r), mob_b, atol=1e-10)


def test_transport_rejects_bad_inputs():
    with pytest.raises(ValueError):
        transport_spatial_mobility(np.eye(3), np.zeros(3))
    with pytest.raises(ValueError):
        transport_spatial_mobility(np.eye(6), np.zeros(2))
