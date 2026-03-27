"""Property-based tests for core mathematical operations.

Uses hypothesis to verify mathematical invariants hold for arbitrary inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from src.shared.python.spatial_algebra.pose6dof import (  # noqa: E402
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
)
from src.shared.python.spatial_algebra.spatial_vectors import skew  # noqa: E402

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

reasonable_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
positive_floats = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)
angles = st.floats(
    min_value=-2 * np.pi, max_value=2 * np.pi, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Rotation matrix properties
# ---------------------------------------------------------------------------


class TestRotationMatrixProperties:
    """Property-based tests for rotation matrices."""

    @given(
        roll=angles,
        pitch=angles,
        yaw=angles,
    )
    @settings(max_examples=100)
    def test_rotation_matrix_is_orthogonal(
        self, roll: float, pitch: float, yaw: float
    ) -> None:
        """Rotation matrices should be orthogonal: R @ R.T = I."""
        R = euler_to_rotation_matrix([roll, pitch, yaw])
        identity = R @ R.T
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)

    @given(
        roll=angles,
        pitch=angles,
        yaw=angles,
    )
    @settings(max_examples=100)
    def test_rotation_matrix_determinant_is_one(
        self, roll: float, pitch: float, yaw: float
    ) -> None:
        """Rotation matrices should have determinant = 1."""
        R = euler_to_rotation_matrix([roll, pitch, yaw])
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    @given(
        roll=st.floats(
            min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False
        ),
        pitch=st.floats(
            min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False
        ),
        yaw=st.floats(
            min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50)
    def test_euler_roundtrip(self, roll: float, pitch: float, yaw: float) -> None:
        """Converting euler->matrix->euler should preserve the rotation (avoiding gimbal lock)."""
        # Avoid gimbal lock region (pitch near +/-pi/2)
        assume(abs(pitch) < 1.4)

        original = np.array([roll, pitch, yaw])
        R = euler_to_rotation_matrix(original)
        recovered = rotation_matrix_to_euler(R)

        # Check that the recovered rotation produces the same matrix
        R_recovered = euler_to_rotation_matrix(recovered)
        np.testing.assert_allclose(R, R_recovered, atol=1e-10)

    @given(
        roll=angles,
        pitch=angles,
        yaw=angles,
    )
    @settings(max_examples=100)
    def test_rotation_preserves_vector_norm(
        self, roll: float, pitch: float, yaw: float
    ) -> None:
        """Rotating a vector should preserve its length."""
        R = euler_to_rotation_matrix([roll, pitch, yaw])
        v = np.array([1.0, 2.0, 3.0])
        v_rotated = R @ v
        np.testing.assert_allclose(
            np.linalg.norm(v_rotated), np.linalg.norm(v), atol=1e-10
        )


# ---------------------------------------------------------------------------
# Skew-symmetric matrix properties
# ---------------------------------------------------------------------------


class TestSkewMatrixProperties:
    """Property-based tests for skew-symmetric matrices."""

    @given(
        x=reasonable_floats,
        y=reasonable_floats,
        z=reasonable_floats,
    )
    @settings(max_examples=100)
    def test_skew_is_antisymmetric(self, x: float, y: float, z: float) -> None:
        """Skew matrix should be antisymmetric: S = -S.T."""
        v = np.array([x, y, z])
        S = skew(v)
        np.testing.assert_allclose(S, -S.T, atol=1e-15)

    @given(
        x=reasonable_floats,
        y=reasonable_floats,
        z=reasonable_floats,
    )
    @settings(max_examples=100)
    def test_skew_diagonal_is_zero(self, x: float, y: float, z: float) -> None:
        """Skew matrix diagonal should be all zeros."""
        v = np.array([x, y, z])
        S = skew(v)
        np.testing.assert_allclose(np.diag(S), [0, 0, 0], atol=1e-15)

    @given(
        x=reasonable_floats,
        y=reasonable_floats,
        z=reasonable_floats,
        ux=reasonable_floats,
        uy=reasonable_floats,
        uz=reasonable_floats,
    )
    @settings(max_examples=100)
    def test_skew_cross_product_equivalence(
        self, x: float, y: float, z: float, ux: float, uy: float, uz: float
    ) -> None:
        """skew(v) @ u should equal cross(v, u)."""
        v = np.array([x, y, z])
        u = np.array([ux, uy, uz])
        S = skew(v)
        # Tolerance scales with input magnitude (float64 has ~15 digits of precision,
        # so products of large values lose precision in the least-significant bits).
        scale = max(np.max(np.abs(v)) * np.max(np.abs(u)), 1.0)
        np.testing.assert_allclose(S @ u, np.cross(v, u), atol=scale * 1e-10)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Property-based tests for numerical stability."""

    @given(
        values=st.lists(
            reasonable_floats,
            min_size=2,
            max_size=100,
        )
    )
    @settings(max_examples=50)
    def test_mean_within_bounds(self, values: list[float]) -> None:
        """Mean of values should be within min/max bounds (up to floating-point rounding)."""
        arr = np.array(values)
        mean = np.mean(arr)
        # Allow a tiny tolerance for floating-point summation rounding
        assert np.min(arr) - 1e-8 <= mean <= np.max(arr) + 1e-8

    @given(
        values=st.lists(
            positive_floats,
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_positive_sum_is_positive(self, values: list[float]) -> None:
        """Sum of positive values should be positive."""
        total = sum(values)
        assert total > 0
