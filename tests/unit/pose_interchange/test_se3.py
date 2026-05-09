"""Unit tests for ``pose_interchange.se3`` helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.se3 import (
    compose_se3,
    euler_xyz_deg_to_matrix,
    inverse_se3,
    is_valid_se3,
    matrix_to_euler_xyz_deg,
    se3_from_xyz_xyz_deg,
    se3_to_xyz_xyz_deg,
)

pytestmark = pytest.mark.unit

_TOL = 1e-9


def _random_safe_euler_deg(rng: np.random.Generator) -> np.ndarray:
    """Random XYZ Euler degrees in a region safely away from y=+-90."""
    return rng.uniform(low=[-170, -75, -170], high=[170, 75, 170])


def test_euler_xyz_matrix_orthonormal() -> None:
    rng = np.random.default_rng(seed=42)
    for _ in range(50):
        euler = _random_safe_euler_deg(rng)
        r = euler_xyz_deg_to_matrix(euler)
        assert r.shape == (3, 3)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=_TOL)
        assert abs(np.linalg.det(r) - 1.0) < _TOL


def test_euler_round_trip_random() -> None:
    rng = np.random.default_rng(seed=7)
    for _ in range(1000):
        euler_in = _random_safe_euler_deg(rng)
        r = euler_xyz_deg_to_matrix(euler_in)
        euler_out = matrix_to_euler_xyz_deg(r)
        np.testing.assert_allclose(euler_out, euler_in, atol=1e-9)


def test_euler_zero_is_identity() -> None:
    r = euler_xyz_deg_to_matrix([0.0, 0.0, 0.0])
    np.testing.assert_allclose(r, np.eye(3), atol=_TOL)


def test_se3_from_to_round_trip() -> None:
    rng = np.random.default_rng(seed=123)
    for _ in range(200):
        t = rng.uniform(-2.0, 2.0, size=3)
        e = _random_safe_euler_deg(rng)
        m = se3_from_xyz_xyz_deg(t, e)
        assert m.shape == (4, 4)
        assert is_valid_se3(m)
        t_back, e_back = se3_to_xyz_xyz_deg(m)
        np.testing.assert_allclose(t_back, t, atol=1e-12)
        np.testing.assert_allclose(e_back, e, atol=1e-9)


def test_compose_and_inverse() -> None:
    rng = np.random.default_rng(seed=4)
    for _ in range(50):
        t = rng.uniform(-1.0, 1.0, size=3)
        e = _random_safe_euler_deg(rng)
        m = se3_from_xyz_xyz_deg(t, e)
        m_inv = inverse_se3(m)
        np.testing.assert_allclose(compose_se3(m, m_inv), np.eye(4), atol=1e-9)
        np.testing.assert_allclose(compose_se3(m_inv, m), np.eye(4), atol=1e-9)


def test_compose_rejects_non_se3() -> None:
    bad = np.eye(4)
    bad[3, 3] = 0.0
    good = se3_from_xyz_xyz_deg([0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="not a valid SE"):
        compose_se3(bad, good)
    with pytest.raises(ValueError, match="not a valid SE"):
        compose_se3(good, bad)


def test_inverse_rejects_non_se3() -> None:
    bad = np.zeros((4, 4))
    with pytest.raises(ValueError, match="not a valid SE"):
        inverse_se3(bad)


def test_se3_from_rejects_bad_translation() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        se3_from_xyz_xyz_deg([1, 2], [0, 0, 0])


def test_se3_from_rejects_bad_rotation() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        se3_from_xyz_xyz_deg([0, 0, 0], [0, 0])


def test_matrix_to_euler_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        matrix_to_euler_xyz_deg(np.eye(4))


def test_se3_to_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        se3_to_xyz_xyz_deg(np.eye(3))


def test_is_valid_se3_basic() -> None:
    assert is_valid_se3(np.eye(4))
    bad = np.eye(4)
    bad[3, 3] = 2.0
    assert not is_valid_se3(bad)
    assert not is_valid_se3(np.zeros((4, 4)))
    assert not is_valid_se3(np.eye(3))
    nan = np.eye(4)
    nan[0, 0] = np.nan
    assert not is_valid_se3(nan)
