"""Edge-case / error-path coverage for ``pose_interchange.se3``.

These tests target previously uncovered branches in
``src/shared/python/pose_interchange/se3.py``:

- ``euler_xyz_deg_to_matrix`` rejecting a wrong-shape rotation vector.
- ``matrix_to_euler_xyz_deg`` taking the gimbal-lock branch when the
  middle rotation is at +/- 90 degrees.
- ``is_valid_se3`` returning False for matrices whose 3x3 block is not
  orthonormal (including a proper shape but non-rotation block) and for
  matrices whose rotation has determinant -1 (improper rotation /
  reflection).

Part of issue #5910 — focused, pure-function coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_interchange.se3 import (
    euler_xyz_deg_to_matrix,
    is_valid_se3,
    matrix_to_euler_xyz_deg,
)

pytestmark = pytest.mark.unit


def test_euler_xyz_deg_to_matrix_rejects_wrong_shape() -> None:
    """A non-(3,) rotation vector must raise ``ValueError``."""
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        euler_xyz_deg_to_matrix([0.0, 0.0])  # type: ignore[list-item]


def test_euler_xyz_deg_to_matrix_rejects_2d_input() -> None:
    """A 2-D array must be rejected with a shape error."""
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        euler_xyz_deg_to_matrix(np.zeros((3, 3)))


def test_matrix_to_euler_gimbal_lock_y_plus_90() -> None:
    """When the middle Y rotation is +90 degrees, the decomposer pins x=0.

    Builds R = Rx(0) @ Ry(90) @ Rz(z0) for a known ``z0``; because of
    gimbal lock the decomposition cannot recover ``(x, y, z)`` uniquely,
    so the implementation folds the rotation into ``z`` with ``x=0``.
    The returned matrix must still round-trip through
    :func:`euler_xyz_deg_to_matrix`.
    """
    z0 = 30.0
    r = euler_xyz_deg_to_matrix([0.0, 90.0, z0])
    out = matrix_to_euler_xyz_deg(r)
    # x is pinned to zero in the gimbal-lock branch.
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    # y is +/- 90; sign should match the input.
    assert out[1] == pytest.approx(90.0, abs=1e-6)
    # The combined (x, z) rotation must reproduce the original matrix.
    r_back = euler_xyz_deg_to_matrix(out)
    np.testing.assert_allclose(r_back, r, atol=1e-7)


def test_matrix_to_euler_gimbal_lock_y_minus_90() -> None:
    """Symmetric gimbal-lock case at y = -90 degrees."""
    r = euler_xyz_deg_to_matrix([0.0, -90.0, 45.0])
    out = matrix_to_euler_xyz_deg(r)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] == pytest.approx(-90.0, abs=1e-6)
    r_back = euler_xyz_deg_to_matrix(out)
    np.testing.assert_allclose(r_back, r, atol=1e-7)


def test_is_valid_se3_rejects_non_orthonormal_rotation() -> None:
    """A 4x4 with the correct last row but a non-orthonormal R is invalid."""
    m = np.eye(4)
    # Scale the rotation block: still has det != 1 and R @ R.T != I.
    m[:3, :3] = np.diag([2.0, 1.0, 1.0])
    assert not is_valid_se3(m)


def test_is_valid_se3_rejects_reflection() -> None:
    """An improper rotation (det == -1) must be rejected.

    The rotation block is orthonormal (R @ R.T == I) but has
    determinant -1, so it is a reflection rather than a proper SE(3)
    rotation. This exercises the final ``abs(det - 1) <= tol`` branch.
    """
    m = np.eye(4)
    # Negate one row to flip the determinant sign while keeping R @ R.T = I.
    m[0, 0] = -1.0
    # Sanity-check the construction: orthonormal but det = -1.
    r = m[:3, :3]
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(-1.0, abs=1e-12)
    assert not is_valid_se3(m)


def test_is_valid_se3_rejects_non_4x4_array() -> None:
    """Wrong-shape inputs return False without raising."""
    assert not is_valid_se3(np.zeros((2, 2)))
    assert not is_valid_se3(np.zeros((4,)))


def test_is_valid_se3_accepts_rotation_built_via_euler() -> None:
    """A matrix built via :func:`euler_xyz_deg_to_matrix` must validate.

    Guards against accidental over-tightening of the orthonormality tolerance.
    """
    m = np.eye(4)
    m[:3, :3] = euler_xyz_deg_to_matrix([10.0, 20.0, 30.0])
    m[:3, 3] = [1.0, -2.0, 3.0]
    assert is_valid_se3(m)
