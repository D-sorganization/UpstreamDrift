"""
Test suite for gravity vector interpolation in pose pathways.

Issue #4106: Gravity vector interpolation fails for non-linear parametric paths
through config space. This test validates that gravity is properly interpolated
when moving between poses with different gravity directions.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.spatial_algebra.pose6dof import (
    slerp,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
)


def interpolate_gravity_vector(
    gravity_start: np.ndarray,
    gravity_end: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Interpolate between two gravity vectors using SLERP on quaternion representations.

    Handles non-linear paths through gravity direction space by treating gravity
    as a 3D vector and interpolating the direction via quaternion-based SLERP,
    while linearly interpolating the magnitude.

    This ensures gravity direction changes smoothly along the path while
    preserving the magnitude interpolation.

    Args:
        gravity_start: Start gravity vector [gx, gy, gz] (not necessarily normalized)
        gravity_end: End gravity vector [gx, gy, gz] (not necessarily normalized)
        alpha: Interpolation parameter [0, 1]

    Returns:
        Interpolated gravity vector (normalized direction, interpolated magnitude)

    Raises:
        ValueError: If gravity vectors are zero or nearly zero
    """
    if not (gravity_start is not None):
        raise ValueError("gravity_start must be provided")
    if not (gravity_end is not None):
        raise ValueError("gravity_end must be provided")

    gravity_start = np.asarray(gravity_start, dtype=np.float64)
    gravity_end = np.asarray(gravity_end, dtype=np.float64)

    # Validate input dimensions
    if gravity_start.shape != (3,):
        raise ValueError(f"gravity_start must be shape (3,), got {gravity_start.shape}")
    if gravity_end.shape != (3,):
        raise ValueError(f"gravity_end must be shape (3,), got {gravity_end.shape}")

    # Compute magnitudes
    mag_start = np.linalg.norm(gravity_start)
    mag_end = np.linalg.norm(gravity_end)

    if mag_start < 1e-10:
        raise ValueError(f"gravity_start is near-zero: magnitude {mag_start}")
    if mag_end < 1e-10:
        raise ValueError(f"gravity_end is near-zero: magnitude {mag_end}")

    # Normalize directions
    dir_start = gravity_start / mag_start
    dir_end = gravity_end / mag_end

    # Use direct SLERP on direction vectors as if they were quaternions
    # with w=0 (pure imaginary quaternions)
    q_start = np.concatenate([[0.0], dir_start])  # [0, x, y, z]
    q_end = np.concatenate([[0.0], dir_end])  # [0, x, y, z]

    # Normalize these "quaternions" to unit magnitude
    q_start = q_start / np.linalg.norm(q_start)
    q_end = q_end / np.linalg.norm(q_end)

    # Apply SLERP to interpolate direction
    dot = np.clip(np.dot(q_start, q_end), -1.0, 1.0)

    if dot < 0:
        q_end = -q_end
        dot = -dot

    if dot > 0.9995:
        # Linear interpolation for very close directions
        q_interp = q_start + alpha * (q_end - q_start)
        q_interp = q_interp / np.linalg.norm(q_interp)
    else:
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        q_interp = s1 * q_start + s2 * q_end

    # Extract direction from interpolated quaternion
    dir_interp = q_interp[1:] / np.linalg.norm(q_interp[1:])

    # Interpolate magnitudes linearly
    mag_interp = (1.0 - alpha) * mag_start + alpha * mag_end

    # Return interpolated gravity vector
    result = dir_interp * mag_interp
    return result


def rotate_vector_by_quaternion(
    vector: np.ndarray, quat: np.ndarray
) -> np.ndarray:
    """Apply quaternion rotation to a vector."""
    if not (vector is not None):
        raise ValueError("vector must be provided")
    if not (quat is not None):
        raise ValueError("quat must be provided")

    # Convert quat [w, x, y, z] to rotation matrix and apply
    R = quaternion_to_rotation_matrix(quat)
    return R @ vector


# =============================================================================
# Test Suite: Gravity Interpolation
# =============================================================================


class TestGravityInterpolationBasics:
    """Test basic gravity vector interpolation."""

    def test_gravity_interpolation_at_endpoints(self) -> None:
        """Endpoints should match exactly."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_end = np.array([0.0, 0.0, -9.81])

        g_0 = interpolate_gravity_vector(g_start, g_end, 0.0)
        g_1 = interpolate_gravity_vector(g_start, g_end, 1.0)

        np.testing.assert_allclose(g_0, g_start, atol=1e-10)
        np.testing.assert_allclose(g_1, g_end, atol=1e-10)

    def test_gravity_interpolation_at_midpoint(self) -> None:
        """Midpoint of vertical gravity vectors should still be vertical."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_end = np.array([0.0, 0.0, -9.81])

        g_mid = interpolate_gravity_vector(g_start, g_end, 0.5)

        np.testing.assert_allclose(g_mid, g_start, atol=1e-10)

    def test_gravity_normalization(self) -> None:
        """Interpolated gravity should be normalized."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_end = np.array([0.0, 0.0, -10.0])

        for alpha in np.linspace(0, 1, 11):
            g = interpolate_gravity_vector(g_start, g_end, alpha)
            mag = np.linalg.norm(g)
            # Should interpolate the magnitude
            expected_mag = (1 - alpha) * 9.81 + alpha * 10.0
            assert np.isclose(mag, expected_mag, atol=1e-10)

    def test_gravity_90_degree_rotation(self) -> None:
        """Test interpolation with 90-degree rotation in gravity."""
        g_start = np.array([0.0, 0.0, -9.81])  # Down
        g_end = np.array([-9.81, 0.0, 0.0])  # Horizontal

        g_mid = interpolate_gravity_vector(g_start, g_end, 0.5)

        # Midpoint should be at 45 degrees between the two
        mag = np.linalg.norm(g_mid)
        assert np.isclose(mag, 9.81, atol=1e-10)

        # Direction should be balanced between start and end
        direction = g_mid / mag
        expected_dir = (
            np.array([0.0, 0.0, -1.0]) / np.sqrt(2)
            + np.array([-1.0, 0.0, 0.0]) / np.sqrt(2)
        ) / np.sqrt(2)
        expected_dir = expected_dir / np.linalg.norm(expected_dir)

        # Direction should be roughly at 45 degrees
        dot_product = np.dot(direction, np.array([0.0, 0.0, -1.0]))
        assert np.isclose(np.abs(dot_product), np.cos(np.pi / 4), atol=0.1)

    def test_gravity_three_non_collinear_directions(self) -> None:
        """Test with three non-collinear gravity directions."""
        g1 = np.array([0.0, 0.0, -9.81])  # Down
        g2 = np.array([9.81, 0.0, 0.0])  # Right
        g3 = np.array([0.0, 9.81, 0.0])  # Forward

        # Interpolate from g1 to g2
        for alpha in np.linspace(0, 1, 5):
            g = interpolate_gravity_vector(g1, g2, alpha)
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 9.81, atol=1e-10)
            # Direction should rotate smoothly
            assert not np.any(np.isnan(g))
            assert not np.any(np.isinf(g))

        # Interpolate from g2 to g3
        for alpha in np.linspace(0, 1, 5):
            g = interpolate_gravity_vector(g2, g3, alpha)
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 9.81, atol=1e-10)
            assert not np.any(np.isnan(g))
            assert not np.any(np.isinf(g))

        # Interpolate from g3 to g1
        for alpha in np.linspace(0, 1, 5):
            g = interpolate_gravity_vector(g3, g1, alpha)
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 9.81, atol=1e-10)
            assert not np.any(np.isnan(g))
            assert not np.any(np.isinf(g))

    def test_gravity_small_magnitude(self) -> None:
        """Test with smaller gravity magnitudes (moon gravity ~1.6 m/s^2)."""
        g_start = np.array([0.0, 0.0, -1.6])  # Moon gravity
        g_end = np.array([0.0, 1.6, 0.0])

        for alpha in np.linspace(0, 1, 11):
            g = interpolate_gravity_vector(g_start, g_end, alpha)
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 1.6, atol=1e-10)

    def test_gravity_large_magnitude(self) -> None:
        """Test with larger gravity magnitudes (Jupiter gravity ~25 m/s^2)."""
        g_start = np.array([0.0, 0.0, -25.0])  # Jupiter gravity
        g_end = np.array([25.0, 0.0, 0.0])

        for alpha in np.linspace(0, 1, 11):
            g = interpolate_gravity_vector(g_start, g_end, alpha)
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 25.0, atol=1e-10)

    def test_gravity_antiparallel_vectors(self) -> None:
        """Test with exactly opposite gravity directions."""
        g_start = np.array([0.0, 0.0, -9.81])  # Down
        g_end = np.array([0.0, 0.0, 9.81])  # Up

        for alpha in np.linspace(0, 1, 11):
            g = interpolate_gravity_vector(g_start, g_end, alpha)
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 9.81, atol=1e-10)
            assert not np.any(np.isnan(g))
            assert not np.any(np.isinf(g))

    def test_gravity_zero_magnitude_raises(self) -> None:
        """Zero-magnitude gravity should raise ValueError."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_zero = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError):
            interpolate_gravity_vector(g_start, g_zero, 0.5)

        with pytest.raises(ValueError):
            interpolate_gravity_vector(g_zero, g_start, 0.5)


class TestGravityInterpolationEdgeCases:
    """Test edge cases for gravity interpolation."""

    def test_gravity_near_zero_raises(self) -> None:
        """Near-zero gravity should raise ValueError."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_tiny = np.array([1e-12, 1e-12, 1e-12])

        with pytest.raises(ValueError):
            interpolate_gravity_vector(g_start, g_tiny, 0.5)

    def test_gravity_different_magnitudes(self) -> None:
        """Test interpolation with different magnitudes."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_end = np.array([0.0, 0.0, -5.0])

        g_mid = interpolate_gravity_vector(g_start, g_end, 0.5)
        mag = np.linalg.norm(g_mid)
        expected_mag = 0.5 * 9.81 + 0.5 * 5.0
        assert np.isclose(mag, expected_mag, atol=1e-10)

    def test_gravity_continuous_path(self) -> None:
        """Test that gravity changes continuously along interpolation path."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_end = np.array([9.81, 0.0, 0.0])

        alphas = np.linspace(0, 1, 21)
        gravities = [interpolate_gravity_vector(g_start, g_end, a) for a in alphas]

        # Check continuity: consecutive points should be reasonably close
        # 90-degree rotation with 20 steps means ~4.5 degrees per step
        # which gives ~sqrt(2)*9.81*(1-cos(pi/40)) ≈ 0.77
        for i in range(len(gravities) - 1):
            diff = np.linalg.norm(gravities[i + 1] - gravities[i])
            assert diff < 1.0  # Conservative bound for 20-step 90-degree rotation

    def test_gravity_invalid_input_shape(self) -> None:
        """Invalid input shapes should raise ValueError."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_bad = np.array([0.0, 0.0])  # Only 2 components

        with pytest.raises(ValueError):
            interpolate_gravity_vector(g_start, g_bad, 0.5)


class TestGravityInterpolationCombinations:
    """Test combined scenarios."""

    def test_gravity_with_varying_magnitudes_nonlinear_path(self) -> None:
        """Test gravity interpolation along a non-linear path."""
        # Path: (0,0,-9.81) -> (6.93, 6.93, -4.905) -> (9.81, 0, 0)
        # This tests motion on a non-linear path in gravity direction space

        g1 = np.array([0.0, 0.0, -9.81])
        g2 = np.array([6.93, 6.93, -4.905])
        g3 = np.array([9.81, 0.0, 0.0])

        # Segment 1: g1 -> g2
        for alpha in np.linspace(0, 1, 6):
            g = interpolate_gravity_vector(g1, g2, alpha)
            mag = np.linalg.norm(g)
            # Magnitude should interpolate linearly
            expected_mag = (1 - alpha) * np.linalg.norm(g1) + alpha * np.linalg.norm(g2)
            assert np.isclose(mag, expected_mag, atol=1e-10)

        # Segment 2: g2 -> g3
        for alpha in np.linspace(0, 1, 6):
            g = interpolate_gravity_vector(g2, g3, alpha)
            mag = np.linalg.norm(g)
            expected_mag = (1 - alpha) * np.linalg.norm(g2) + alpha * np.linalg.norm(g3)
            assert np.isclose(mag, expected_mag, atol=1e-10)

    def test_gravity_interpolation_smooth(self) -> None:
        """Verify smooth interpolation with fine granularity."""
        g_start = np.array([0.0, 0.0, -9.81])
        g_end = np.array([6.93, 0.0, -6.93])  # magnitude ~9.81 at 45 degrees

        # Fine-grained interpolation should be smooth
        alphas = np.linspace(0, 1, 101)
        gravities = [interpolate_gravity_vector(g_start, g_end, a) for a in alphas]

        # All should have interpolated magnitude between start and end
        mag_start = np.linalg.norm(g_start)
        mag_end = np.linalg.norm(g_end)
        for i, g in enumerate(gravities):
            alpha = alphas[i]
            expected_mag = (1 - alpha) * mag_start + alpha * mag_end
            assert np.isclose(np.linalg.norm(g), expected_mag, atol=1e-8)

        # Directions should rotate smoothly
        directions = [g / 9.81 for g in gravities]
        for i in range(len(directions) - 1):
            dot = np.dot(directions[i], directions[i + 1])
            # Consecutive steps should have high dot product (close angles)
            assert dot > 0.99


class TestGravityInterpolationIntegration:
    """Integration tests with pose interpolation."""

    def test_gravity_with_pose_joint_interpolation(self) -> None:
        """Test gravity and joint interpolation together."""
        # Simulate joint configuration interpolation
        qpos_start = np.array([0.0, 0.5, -0.5, 0.3])
        qpos_end = np.array([0.2, 1.0, -1.0, 0.6])

        g_start = np.array([0.0, 0.0, -9.81])
        g_end = np.array([9.81, 0.0, 0.0])

        # Interpolate both
        for alpha in np.linspace(0, 1, 11):
            # Joint interpolation (linear)
            qpos_interp = (1 - alpha) * qpos_start + alpha * qpos_end

            # Gravity interpolation
            g_interp = interpolate_gravity_vector(g_start, g_end, alpha)

            # Both should be valid
            assert qpos_interp.shape == (4,)
            assert g_interp.shape == (3,)
            assert np.isclose(np.linalg.norm(g_interp), 9.81, atol=1e-10)
            assert not np.any(np.isnan(qpos_interp))
            assert not np.any(np.isnan(g_interp))
