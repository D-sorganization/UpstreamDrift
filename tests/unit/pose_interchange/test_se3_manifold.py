"""Tests for the canonical-v2 quaternion + manifold helpers in se3.py (CC-2).

Contract: docs/conventions/canonical-v2.md (ADR-0026). Quaternions are
unit, scalar-first ``(w, x, y, z)``. ``quat_exp`` maps a rotation vector
(axis * angle, radians) to a unit quaternion; ``quat_log`` is its inverse
on the principal branch ``||rotvec|| < pi``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.shared.python.pose_interchange import se3

pytestmark = pytest.mark.unit

_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _rotvecs(max_norm: float = 3.0):
    """Hypothesis strategy yielding rotation vectors with norm < max_norm."""
    comp = st.floats(
        min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
    )
    return (
        st.tuples(comp, comp, comp)
        .map(np.array)
        .filter(lambda v: np.linalg.norm(v) < max_norm)
    )


# ---- quat_exp / quat_log -----------------------------------------------------


def test_quat_exp_zero_is_identity() -> None:
    assert np.allclose(se3.quat_exp(np.zeros(3)), _IDENTITY)


def test_quat_log_identity_is_zero() -> None:
    assert np.allclose(se3.quat_log(_IDENTITY), np.zeros(3))


def test_quat_exp_is_unit_norm() -> None:
    q = se3.quat_exp(np.array([0.3, -1.2, 0.7]))
    assert np.isclose(np.linalg.norm(q), 1.0)


@given(_rotvecs(max_norm=np.pi - 1e-3))
@settings(max_examples=200)
def test_quat_log_exp_roundtrip(rotvec: np.ndarray) -> None:
    """quat_log(quat_exp(v)) == v on the principal branch."""
    recovered = se3.quat_log(se3.quat_exp(rotvec))
    assert np.allclose(recovered, rotvec, atol=1e-9)


@given(_rotvecs(max_norm=np.pi - 1e-3))
@settings(max_examples=200)
def test_quat_exp_log_roundtrip_as_rotation(rotvec: np.ndarray) -> None:
    """quat_exp(quat_log(q)) == q (up to sign) — compare as rotation matrices."""
    q = se3.quat_exp(rotvec)
    q2 = se3.quat_exp(se3.quat_log(q))
    assert np.allclose(se3.quat_to_matrix(q), se3.quat_to_matrix(q2), atol=1e-9)


def test_quat_log_canonicalizes_sign() -> None:
    """q and -q are the same rotation; quat_log returns the same minimal rotvec."""
    q = se3.quat_exp(np.array([0.1, 0.2, -0.3]))
    assert np.allclose(se3.quat_log(q), se3.quat_log(-q), atol=1e-9)


def test_quat_log_large_rotation_near_pi() -> None:
    rotvec = np.array([np.pi - 1e-4, 0.0, 0.0])
    assert np.allclose(se3.quat_log(se3.quat_exp(rotvec)), rotvec, atol=1e-6)


# ---- quat algebra ------------------------------------------------------------


def test_quat_multiply_identity() -> None:
    q = se3.quat_exp(np.array([0.5, -0.2, 0.9]))
    assert np.allclose(se3.quat_multiply(q, _IDENTITY), q)
    assert np.allclose(se3.quat_multiply(_IDENTITY, q), q)


def test_quat_conjugate_is_inverse() -> None:
    q = se3.quat_exp(np.array([0.4, 0.1, -0.6]))
    prod = se3.quat_multiply(q, se3.quat_conjugate(q))
    # identity up to sign
    assert np.allclose(np.abs(prod), np.abs(_IDENTITY), atol=1e-12)
    assert np.isclose(prod[0], 1.0, atol=1e-12)


@given(_rotvecs(), _rotvecs())
@settings(max_examples=100)
def test_quat_multiply_matches_matrix_product(a: np.ndarray, b: np.ndarray) -> None:
    """Composing quaternions equals composing their rotation matrices."""
    qa, qb = se3.quat_exp(a), se3.quat_exp(b)
    lhs = se3.quat_to_matrix(se3.quat_multiply(qa, qb))
    rhs = se3.quat_to_matrix(qa) @ se3.quat_to_matrix(qb)
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_quat_normalize_rejects_zero() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        se3.quat_normalize(np.zeros(4))


def test_quat_to_matrix_orthonormal() -> None:
    q = se3.quat_exp(np.array([0.7, 1.1, -0.4]))
    r = se3.quat_to_matrix(q)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)


# ---- v1 -> v2 migration helper ----------------------------------------------


@given(
    st.tuples(
        st.floats(min_value=-80, max_value=80, allow_nan=False),
        st.floats(min_value=-80, max_value=80, allow_nan=False),
        st.floats(min_value=-80, max_value=80, allow_nan=False),
    ).map(np.array)
)
@settings(max_examples=100)
def test_euler_xyz_deg_to_quat_matches_matrix(euler_deg: np.ndarray) -> None:
    """The v1 Euler-degrees pose maps to a quaternion with the same rotation."""
    q = se3.euler_xyz_deg_to_quat_wxyz(euler_deg)
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(
        se3.quat_to_matrix(q), se3.euler_xyz_deg_to_matrix(euler_deg), atol=1e-9
    )
