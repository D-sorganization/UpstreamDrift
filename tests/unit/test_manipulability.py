"""Tests for manipulability and Jacobian conditioning module.

Tests the Jacobian condition number monitoring and manipulability analysis
that implements Guideline C2 for singularity detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.manipulability import (
    CATASTROPHIC_SINGULARITY_THRESHOLD,
    SINGULARITY_FALLBACK_THRESHOLD,
    SINGULARITY_WARNING_THRESHOLD,
    SingularityError,
    check_jacobian_conditioning,
    compute_manipulability_ellipsoid,
    compute_manipulability_index,
    get_jacobian_conditioning,
)


class MockEngine:
    """Mock physics engine for testing."""

    def __init__(self, jacobian_dict=None):
        """Initialize mock engine with optional jacobian dict."""
        self.jacobian_dict = jacobian_dict

    def compute_jacobian(self, body_name):
        """Return mock Jacobian."""
        return self.jacobian_dict


class TestCheckJacobianConditioning:
    """Test check_jacobian_conditioning function."""

    def test_well_conditioned_jacobian(self):
        """Test with well-conditioned Jacobian (κ < 1e6)."""
        # Identity matrix has κ = 1.0
        J = np.eye(3)

        kappa = check_jacobian_conditioning(J, "test_body", warn=False)

        assert kappa == pytest.approx(1.0, rel=1e-10)

    def test_near_singularity_warning(self, caplog):
        """Test warning at near-singularity (κ > 1e6)."""
        # Create matrix with κ ≈ 1e7
        J = np.diag([1.0, 1.0, 1e-7])

        with caplog.at_level("WARNING"):
            kappa = check_jacobian_conditioning(J, "test_body", warn=True)

        assert kappa > SINGULARITY_WARNING_THRESHOLD
        assert "Near-singularity" in caplog.text

    def test_severe_ill_conditioning(self, caplog):
        """Test error logging at severe ill-conditioning (κ > 1e10)."""
        # Create matrix with κ ≈ 1e11
        J = np.diag([1.0, 1.0, 1e-11])

        with caplog.at_level("ERROR"):
            kappa = check_jacobian_conditioning(J, "test_body", warn=True)

        assert kappa > SINGULARITY_FALLBACK_THRESHOLD
        assert "SEVERE ILL-CONDITIONING" in caplog.text

    def test_catastrophic_singularity_raises_error(self):
        """Test that catastrophic singularity raises SingularityError."""
        # Create matrix with κ > 1e12
        J = np.diag([1.0, 1.0, 1e-13])

        with pytest.raises(SingularityError, match="CATASTROPHIC SINGULARITY"):
            check_jacobian_conditioning(J, "test_body", warn=True)

    def test_warning_suppression(self, caplog):
        """Test that warnings can be suppressed with warn=False."""
        J = np.diag([1.0, 1.0, 1e-7])  # κ > 1e6

        with caplog.at_level("WARNING"):
            kappa = check_jacobian_conditioning(J, "test_body", warn=False)

        # Should still return kappa but not log warning
        assert kappa > SINGULARITY_WARNING_THRESHOLD
        # No warning should be logged
        singularity_warnings = [r for r in caplog.records if "singularity" in r.message.lower()]
        assert len(singularity_warnings) == 0

    def test_empty_jacobian(self, caplog):
        """Test handling of empty Jacobian."""
        J = np.array([])

        with caplog.at_level("WARNING"):
            kappa = check_jacobian_conditioning(J, "test_body", warn=True)

        assert np.isinf(kappa)
        assert "Empty Jacobian" in caplog.text

    def test_zero_row_jacobian(self, caplog):
        """Test handling of Jacobian with zero rows."""
        J = np.zeros((0, 3))

        with caplog.at_level("WARNING"):
            kappa = check_jacobian_conditioning(J, "test_body", warn=True)

        assert np.isinf(kappa)

    def test_zero_column_jacobian(self, caplog):
        """Test handling of Jacobian with zero columns."""
        J = np.zeros((3, 0))

        with caplog.at_level("WARNING"):
            kappa = check_jacobian_conditioning(J, "test_body", warn=True)

        assert np.isinf(kappa)

    def test_rectangular_jacobian(self):
        """Test with rectangular Jacobian (typical 6×n)."""
        # 6 rows (spatial) × 3 cols (joints)
        J = np.random.rand(6, 3)

        kappa = check_jacobian_conditioning(J, "test_body", warn=False)

        assert kappa > 0
        assert np.isfinite(kappa)

    def test_body_name_in_logs(self, caplog):
        """Test that body name appears in log messages."""
        J = np.diag([1.0, 1e-7])  # Near-singular

        with caplog.at_level("WARNING"):
            check_jacobian_conditioning(J, "my_special_body", warn=True)

        assert "my_special_body" in caplog.text


class TestGetJacobianConditioning:
    """Test get_jacobian_conditioning function."""

    def test_with_spatial_jacobian(self):
        """Test retrieval with spatial Jacobian."""
        J_spatial = np.eye(6)
        engine = MockEngine(jacobian_dict={"spatial": J_spatial})

        kappa = get_jacobian_conditioning(engine, "test_body", warn=False)

        assert kappa == pytest.approx(1.0, rel=1e-10)

    def test_with_linear_jacobian_only(self):
        """Test fallback to linear Jacobian when spatial not available."""
        J_linear = np.eye(3)
        engine = MockEngine(jacobian_dict={"linear": J_linear})

        kappa = get_jacobian_conditioning(engine, "test_body", warn=False)

        assert kappa == pytest.approx(1.0, rel=1e-10)

    def test_prefers_spatial_over_linear(self):
        """Test that spatial Jacobian is preferred when both exist."""
        J_spatial = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])  # κ = 2
        J_linear = np.diag([1.0, 1.0, 3.0])  # κ = 3
        engine = MockEngine(jacobian_dict={"spatial": J_spatial, "linear": J_linear})

        kappa = get_jacobian_conditioning(engine, "test_body", warn=False)

        # Should use spatial (κ=2) not linear (κ=3)
        assert kappa == pytest.approx(2.0, rel=1e-6)

    def test_body_not_found(self, caplog):
        """Test handling when body not found in model."""
        engine = MockEngine(jacobian_dict=None)

        with caplog.at_level("WARNING"):
            kappa = get_jacobian_conditioning(engine, "nonexistent_body", warn=False)

        assert np.isinf(kappa)
        assert "not found" in caplog.text

    def test_empty_jacobian_dict(self, caplog):
        """Test handling of empty Jacobian dictionary."""
        engine = MockEngine(jacobian_dict={})

        with caplog.at_level("WARNING"):
            kappa = get_jacobian_conditioning(engine, "test_body", warn=False)

        assert np.isinf(kappa)
        assert "No Jacobian data" in caplog.text


class TestComputeManipulabilityEllipsoid:
    """Test compute_manipulability_ellipsoid function."""

    def test_identity_matrix(self):
        """Test ellipsoid for identity Jacobian."""
        J = np.eye(3)

        radii, axes = compute_manipulability_ellipsoid(J)

        # All singular values should be 1.0
        np.testing.assert_allclose(radii, [1.0, 1.0, 1.0], rtol=1e-10)

        # Axes should form orthonormal basis
        assert axes.shape == (3, 3)

    def test_diagonal_matrix(self):
        """Test ellipsoid for diagonal Jacobian."""
        J = np.diag([3.0, 2.0, 1.0])

        radii, axes = compute_manipulability_ellipsoid(J)

        # Radii should be sorted in descending order
        expected_radii = np.array([3.0, 2.0, 1.0])
        np.testing.assert_allclose(radii, expected_radii, rtol=1e-10)

    def test_rectangular_jacobian(self):
        """Test with rectangular matrix (6×3)."""
        J = np.random.rand(6, 3)

        radii, axes = compute_manipulability_ellipsoid(J)

        # Should have 3 radii (min dimension)
        assert radii.shape == (3,)

        # Axes should be (3, 3)
        assert axes.shape == (3, 3)

        # All radii should be non-negative
        assert np.all(radii >= 0)

    def test_axes_orthonormal(self):
        """Test that principal axes are orthonormal."""
        J = np.random.rand(5, 4)

        radii, axes = compute_manipulability_ellipsoid(J)

        # Check orthonormality: V^T V = I (with numerical tolerance)
        should_be_identity = axes.T @ axes
        np.testing.assert_allclose(should_be_identity, np.eye(4), rtol=1e-6, atol=1e-12)

    def test_radii_sorted_descending(self):
        """Test that radii are sorted in descending order."""
        J = np.diag([5.0, 3.0, 7.0, 1.0])

        radii, axes = compute_manipulability_ellipsoid(J)

        # SVD returns singular values in descending order
        assert np.all(radii[:-1] >= radii[1:])

    def test_returns_tuple(self):
        """Test that function returns a tuple of two arrays."""
        J = np.random.rand(3, 3)

        result = compute_manipulability_ellipsoid(J)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)


class TestComputeManipulabilityIndex:
    """Test compute_manipulability_index function."""

    def test_identity_matrix(self):
        """Test manipulability index for identity matrix."""
        J = np.eye(3)

        mu = compute_manipulability_index(J)

        # Product of singular values (all 1.0) = 1.0
        assert mu == pytest.approx(1.0, rel=1e-10)

    def test_diagonal_matrix(self):
        """Test manipulability index for diagonal matrix."""
        J = np.diag([2.0, 3.0, 4.0])

        mu = compute_manipulability_index(J)

        # μ = product of singular values = 2 × 3 × 4 = 24
        expected_mu = 2.0 * 3.0 * 4.0
        assert mu == pytest.approx(expected_mu, rel=1e-10)

    def test_singular_matrix_zero_index(self):
        """Test that singular matrix has zero manipulability index."""
        # Singular matrix (rank deficient)
        J = np.array([[1, 2], [2, 4]])  # Second row is 2x first row

        mu = compute_manipulability_index(J)

        # Should be approximately zero (within numerical precision)
        assert mu == pytest.approx(0.0, abs=1e-10)

    def test_near_singular_small_index(self):
        """Test that near-singular matrix has small manipulability index."""
        J = np.diag([1.0, 1.0, 1e-10])

        mu = compute_manipulability_index(J)

        # μ = 1 × 1 × 1e-10 = 1e-10
        assert mu < 1e-9

    def test_well_conditioned_large_index(self):
        """Test that well-conditioned matrix has reasonable index."""
        J = np.diag([5.0, 5.0, 5.0])

        mu = compute_manipulability_index(J)

        # μ = 5 × 5 × 5 = 125
        assert mu == pytest.approx(125.0, rel=1e-10)

    def test_rectangular_matrix(self):
        """Test with rectangular Jacobian (6×3)."""
        J = np.random.rand(6, 3)

        mu = compute_manipulability_index(J)

        # Should be finite and non-negative
        assert np.isfinite(mu)
        assert mu >= 0

    def test_return_type_is_float(self):
        """Test that return type is Python float."""
        J = np.eye(3)

        mu = compute_manipulability_index(J)

        assert isinstance(mu, float)

    def test_yoshikawa_formula(self):
        """Test that index matches Yoshikawa's formula: μ = √det(J J^T)."""
        J = np.array([[1, 0], [0, 2], [0, 0]])  # 3×2 matrix

        mu = compute_manipulability_index(J)

        # Yoshikawa: μ = √det(J @ J^T)
        # But SVD approach: μ = ∏ σ_i
        # For this matrix, singular values are [2, 1], so μ = 2
        expected_mu = 2.0
        assert mu == pytest.approx(expected_mu, rel=1e-10)


class TestSingularityThresholds:
    """Test that singularity thresholds are correctly defined."""

    def test_thresholds_are_defined(self):
        """Test that all threshold constants are defined."""
        assert SINGULARITY_WARNING_THRESHOLD == 1e6
        assert SINGULARITY_FALLBACK_THRESHOLD == 1e10
        assert CATASTROPHIC_SINGULARITY_THRESHOLD == 1e12

    def test_thresholds_ordered_correctly(self):
        """Test that thresholds are in ascending order."""
        assert SINGULARITY_WARNING_THRESHOLD < SINGULARITY_FALLBACK_THRESHOLD
        assert SINGULARITY_FALLBACK_THRESHOLD < CATASTROPHIC_SINGULARITY_THRESHOLD


class TestSingularityError:
    """Test SingularityError exception."""

    def test_exception_can_be_raised(self):
        """Test that SingularityError can be raised."""
        with pytest.raises(SingularityError):
            raise SingularityError("Test error")

    def test_exception_inherits_from_exception(self):
        """Test that SingularityError is an Exception."""
        assert issubclass(SingularityError, Exception)

    def test_exception_message_preserved(self):
        """Test that exception message is preserved."""
        msg = "Critical singularity detected"
        try:
            raise SingularityError(msg)
        except SingularityError as e:
            assert str(e) == msg


class TestPhysicalRealism:
    """Test physical realism and practical use cases."""

    def test_fully_extended_arm_high_condition_number(self):
        """Test that fully extended configuration has high condition number.

        When a robotic arm is fully extended, the Jacobian becomes
        near-singular (small manipulability in extension direction).
        """
        # Simplified 2-link planar arm at full extension
        # J = [[-L1-L2, -L2], [0, 0]]  (y-velocity is zero)
        L1, L2 = 1.0, 0.5
        J = np.array([
            [-L1 - L2, -L2],
            [1e-10, 1e-10]  # Near-zero, simulating extension
        ])

        kappa = check_jacobian_conditioning(J, "extended_arm", warn=False)

        # Should have high condition number
        assert kappa > 1e6

    def test_optimal_configuration_low_condition_number(self):
        """Test that well-posed configuration has low condition number."""
        # Jacobian at optimal configuration (well-conditioned)
        J = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])

        kappa = check_jacobian_conditioning(J, "optimal_config", warn=False)

        # Should have low condition number (well-conditioned)
        assert kappa < 10

    def test_gimbal_lock_singularity(self):
        """Test detection of gimbal lock configuration."""
        # At gimbal lock, one DOF is lost (two axes align)
        # Use near-singular but not catastrophic matrix
        J = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1e-8]  # Nearly aligned with second, but not exactly
        ])

        kappa = check_jacobian_conditioning(J, "gimbal_lock", warn=False)

        # Should detect singularity (high condition number)
        assert kappa > SINGULARITY_WARNING_THRESHOLD


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_column_jacobian(self):
        """Test with single column (1-DOF system)."""
        J = np.array([[1.0], [2.0], [3.0]])

        radii, axes = compute_manipulability_ellipsoid(J)

        assert radii.shape == (1,)
        assert axes.shape == (1, 1)

    def test_single_row_jacobian(self):
        """Test with single row."""
        J = np.array([[1.0, 2.0, 3.0]])

        radii, axes = compute_manipulability_ellipsoid(J)

        assert radii.shape == (1,)
        # axes.shape should be (3, 1) from SVD of (1, 3) matrix
        assert axes.shape == (3, 1)

    def test_very_large_jacobian_values(self):
        """Test with large values in Jacobian."""
        J = np.eye(3) * 1e5

        mu = compute_manipulability_index(J)

        # μ = 1e5 × 1e5 × 1e5 = 1e15
        assert mu == pytest.approx(1e15, rel=1e-6)

    def test_very_small_jacobian_values(self):
        """Test with small values in Jacobian."""
        J = np.eye(3) * 1e-5

        mu = compute_manipulability_index(J)

        # μ = 1e-5 × 1e-5 × 1e-5 = 1e-15
        assert mu == pytest.approx(1e-15, rel=1e-6)

    def test_mixed_scale_jacobian(self):
        """Test with widely varying scales in Jacobian."""
        J = np.diag([1e-3, 1.0, 1e3])

        kappa = check_jacobian_conditioning(J, "mixed_scale", warn=False)

        # κ = 1e3 / 1e-3 = 1e6
        assert kappa == pytest.approx(1e6, rel=1e-3)
