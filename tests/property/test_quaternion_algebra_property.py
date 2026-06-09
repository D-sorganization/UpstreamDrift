"""Property-based tests for quaternion algebra (issue #7132).

The A-O test-hardening audit noted that property-based tests are rare relative
to the amount of numerical logic. The euler<->matrix path already has property
coverage (tests/unit/test_property_based.py), but the quaternion algebra in
``src.shared.python.spatial_algebra.pose6dof.rotations`` did not. These tests
add adversarial invariant coverage: Hamilton-product identities, inverse
involution, the rotation homomorphism (R(p*q) == R(p) @ R(q)), and the
quaternion<->rotation-matrix round trip (up to the q ~ -q double cover).
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402

from src.shared.python.spatial_algebra.pose6dof import (  # noqa: E402
    axis_angle_to_rotation_matrix,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

pytestmark = pytest.mark.unit

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

_quat_components = hnp.arrays(
    dtype=np.float64,
    shape=(4,),
    elements=st.floats(
        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
)

_unit_axis = hnp.arrays(
    dtype=np.float64,
    shape=(3,),
    elements=st.floats(
        min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)

_angles = st.floats(min_value=-np.pi + 1e-3, max_value=np.pi - 1e-3, allow_nan=False)


def _normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


@given(_quat_components)
@settings(max_examples=200, deadline=None)
def test_multiply_by_identity_is_identity(q: np.ndarray) -> None:
    assume(np.linalg.norm(q) > 1e-3)
    q = _normalize(q)
    left = quaternion_multiply(_IDENTITY_QUAT, q)
    right = quaternion_multiply(q, _IDENTITY_QUAT)
    np.testing.assert_allclose(left, q, atol=1e-9)
    np.testing.assert_allclose(right, q, atol=1e-9)


@given(_quat_components)
@settings(max_examples=200, deadline=None)
def test_inverse_yields_identity_rotation(q: np.ndarray) -> None:
    assume(np.linalg.norm(q) > 1e-3)
    q = _normalize(q)
    product = quaternion_multiply(q, quaternion_inverse(q))
    # q * q^-1 is the identity rotation: real part +/-1, vector part ~0.
    np.testing.assert_allclose(product[1:], np.zeros(3), atol=1e-8)
    assert abs(abs(product[0]) - 1.0) < 1e-8


@given(_quat_components, _quat_components, _quat_components)
@settings(max_examples=150, deadline=None)
def test_multiplication_is_associative(
    p: np.ndarray, q: np.ndarray, r: np.ndarray
) -> None:
    for value in (p, q, r):
        assume(np.linalg.norm(value) > 1e-3)
    p, q, r = _normalize(p), _normalize(q), _normalize(r)
    left = quaternion_multiply(quaternion_multiply(p, q), r)
    right = quaternion_multiply(p, quaternion_multiply(q, r))
    np.testing.assert_allclose(left, right, atol=1e-8)


@given(_quat_components, _quat_components)
@settings(max_examples=150, deadline=None)
def test_rotation_homomorphism(p: np.ndarray, q: np.ndarray) -> None:
    """R(p * q) must equal R(p) @ R(q) (group homomorphism)."""
    assume(np.linalg.norm(p) > 1e-3)
    assume(np.linalg.norm(q) > 1e-3)
    p, q = _normalize(p), _normalize(q)
    composed = quaternion_to_rotation_matrix(quaternion_multiply(p, q))
    chained = quaternion_to_rotation_matrix(p) @ quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(composed, chained, atol=1e-8)


@given(_quat_components)
@settings(max_examples=200, deadline=None)
def test_quaternion_matrix_round_trip(q: np.ndarray) -> None:
    """quat -> R -> quat preserves the rotation (modulo the q ~ -q sign)."""
    assume(np.linalg.norm(q) > 1e-3)
    q = _normalize(q)
    recovered = rotation_matrix_to_quaternion(quaternion_to_rotation_matrix(q))
    recovered = _normalize(recovered)
    # Double cover: q and -q encode the same rotation, so compare R again.
    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(recovered),
        quaternion_to_rotation_matrix(q),
        atol=1e-7,
    )


@given(_unit_axis, _angles)
@settings(max_examples=150, deadline=None)
def test_axis_angle_matrix_is_orthonormal_rotation(
    axis: np.ndarray, angle: float
) -> None:
    """axis_angle_to_rotation_matrix must yield a proper rotation (R^T R = I,
    det = +1) and survive the matrix<->quaternion round trip."""
    assume(np.linalg.norm(axis) > 1e-2)
    rotation = axis_angle_to_rotation_matrix(axis, angle)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-7
    round_trip = quaternion_to_rotation_matrix(rotation_matrix_to_quaternion(rotation))
    np.testing.assert_allclose(round_trip, rotation, atol=1e-6)
