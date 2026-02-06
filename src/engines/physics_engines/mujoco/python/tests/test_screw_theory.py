"""
Unit tests for screw theory module.

Tests exponential/logarithmic maps, screw axes, and adjoint transforms.
"""

import numpy as np
import pytest
from mujoco_humanoid_golf.screw_theory import (
    adjoint_transform,
    exponential_map,
    logarithmic_map,
    screw_axis,
    screw_to_transform,
    twist_to_spatial,
    wrench_to_spatial,
)


class TestTwistsAndWrenches:
    """Tests for twist and wrench conversions."""

    def test_twist_to_spatial_basic(self) -> None:
        """Test basic twist conversion."""
        omega = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
        V = twist_to_spatial(omega, v)

        np.testing.assert_allclose(V[:3], omega, atol=1e-10)
        np.testing.assert_allclose(V[3:], v, atol=1e-10)

    def test_twist_to_spatial_with_point(self) -> None:
        """Test twist conversion with reference point."""
        omega = np.array([0, 0, 1])  # Rotation about z
        v = np.array([0, 0, 0])  # Zero velocity at origin
        point = np.array([1, 0, 0])  # Point at (1, 0, 0)
        V = twist_to_spatial(omega, v, point)

        # Should have linear velocity from rotation
        assert abs(V[3]) > 0 or abs(V[4]) > 0  # Some linear component

    def test_wrench_to_spatial_basic(self) -> None:
        """Test basic wrench conversion."""
        moment = np.array([0, 0, 1])
        force = np.array([10, 0, 0])
        F = wrench_to_spatial(moment, force)

        np.testing.assert_allclose(F[:3], moment, atol=1e-10)
        np.testing.assert_allclose(F[3:], force, atol=1e-10)

    def test_wrench_to_spatial_with_point(self) -> None:
        """Test wrench conversion with reference point."""
        moment = np.array([0, 0, 0])
        force = np.array([10, 0, 0])  # Force along x
        point = np.array([0, 1, 0])  # Point at (0, 1, 0)
        F = wrench_to_spatial(moment, force, point)

        # Should have moment from force at offset point
        assert abs(F[2]) > 0  # Moment about z


class TestScrewAxes:
    """Tests for screw axis representations."""

    def test_screw_axis_pure_rotation(self) -> None:
        """Test screw axis for pure rotation."""
        axis = np.array([0, 0, 1])  # z-axis
        point = np.array([0, 0, 0])  # origin
        S = screw_axis(axis, point, 0)

        expected = np.array([0, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(S, expected, atol=1e-10)

    def test_screw_axis_pure_translation(self) -> None:
        """Test screw axis for pure translation."""
        axis = np.array([1, 0, 0])
        point = np.array([0, 0, 0])
        S = screw_axis(axis, point, np.inf)

        expected = np.array([0, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(S, expected, atol=1e-10)

    def test_screw_axis_normalization(self) -> None:
        """Test screw axis normalizes direction vector."""
        axis = np.array([2, 0, 0])  # Not unit length
        point = np.array([0, 0, 0])
        S = screw_axis(axis, point, 0)

        # Angular part should be normalized
        assert np.isclose(np.linalg.norm(S[:3]), 1.0)

    def test_screw_axis_with_pitch(self) -> None:
        """Test screw axis with non-zero pitch."""
        axis = np.array([0, 0, 1])
        point = np.array([0, 0, 0])
        pitch = 0.1  # Non-zero pitch
        S = screw_axis(axis, point, pitch)

        # Should have both angular and linear components
        assert np.isclose(np.linalg.norm(S[:3]), 1.0)
        assert abs(S[5]) > 0  # Linear component along z


class TestExponentialMap:
    """Tests for exponential map."""

    def test_exponential_pure_rotation(self) -> None:
        """Test exponential map for pure rotation."""
        S = np.array([0, 0, 1, 0, 0, 0])  # Rotation about z
        theta = np.pi / 2  # 90 degrees

        T = exponential_map(S, theta)

        # Expected rotation matrix
        R_expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        np.testing.assert_allclose(T[:3, :3], R_expected, atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], 0, atol=1e-10)

    def test_exponential_pure_translation(self) -> None:
        """Test exponential map for pure translation."""
        S = np.array([0, 0, 0, 1, 0, 0])  # Translation along x
        theta = 1.5

        T = exponential_map(S, theta)

        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)

        p_expected = np.array([1.5, 0, 0])
        np.testing.assert_allclose(T[:3, 3], p_expected, atol=1e-10)

    def test_exponential_homogeneous_row(self) -> None:
        """Test bottom row is [0 0 0 1]."""
        S = np.array([0, 0, 1, 0, 0, 0])
        theta = 0.5

        T = exponential_map(S, theta)

        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)


class TestLogarithmicMap:
    """Tests for logarithmic map."""

    def test_logarithmic_inverse_of_exponential(self) -> None:
        """Test logarithmic map inverts exponential map."""
        S_orig = screw_axis(np.array([0, 0, 1]), np.array([1, 0, 0]), 0)
        theta_orig = np.pi / 3

        # Apply exponential map
        T = exponential_map(S_orig, theta_orig)

        # Apply logarithmic map
        S_recovered, theta_recovered = logarithmic_map(T)

        # S*theta should match (S can differ by sign)
        screw_orig = S_orig * theta_orig
        screw_recovered = S_recovered * theta_recovered

        np.testing.assert_allclose(screw_orig, screw_recovered, atol=1e-8)

    def test_logarithmic_pure_translation(self) -> None:
        """Test logarithmic map for pure translation."""
        T = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        S, theta = logarithmic_map(T)

        # Should detect pure translation
        assert np.isclose(np.linalg.norm(S[:3]), 0)
        assert theta > 0

    def test_logarithmic_identity(self) -> None:
        """Test logarithmic map for identity transformation."""
        T = np.eye(4)
        S, theta = logarithmic_map(T)

        # For identity transformation, should return zero screw axis and
        # zero displacement
        assert np.allclose(
            S,
            0,
            atol=1e-10,
        ), f"Identity transformation should have zero screw axis, got S={S}"
        assert (
            abs(theta) < 1e-10
        ), f"Identity transformation should have zero displacement, got theta={theta}"


class TestAdjointTransform:
    """Tests for adjoint transformations."""

    def test_adjoint_shape(self) -> None:
        """Test adjoint has correct shape."""
        T = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])

        Ad = adjoint_transform(T)

        assert Ad.shape == (6, 6)

    def test_adjoint_structure(self) -> None:
        """Test adjoint has correct block structure."""
        T = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])

        Ad = adjoint_transform(T)

        # Top-right should be zero (assuming [omega; v] twist order)
        # This matches Featherstone's spatial vector convention used elsewhere
        np.testing.assert_allclose(Ad[:3, 3:], 0, atol=1e-10)

        # Bottom-left should NOT be zero for this transformation
        # (p = [1, 2, 3], so skew(p) is non-zero)
        assert not np.allclose(Ad[3:, :3], 0, atol=1e-10)

    def test_adjoint_composition_property(self) -> None:
        """Test Ad(T1*T2) = Ad(T1) * Ad(T2)."""
        T1 = exponential_map(np.array([0, 0, 1, 0, 0, 0]), np.pi / 4)
        T2 = exponential_map(np.array([0, 0, 0, 1, 0, 0]), 0.5)

        Ad1 = adjoint_transform(T1)
        Ad2 = adjoint_transform(T2)
        Ad_comp = adjoint_transform(T1 @ T2)

        np.testing.assert_allclose(Ad_comp, Ad1 @ Ad2, atol=1e-10)

    def test_adjoint_identity(self) -> None:
        """Test adjoint of identity is identity."""
        T = np.eye(4)
        Ad = adjoint_transform(T)

        np.testing.assert_allclose(Ad, np.eye(6), atol=1e-10)


class TestScrewToTransform:
    """Tests for screw_to_transform convenience function."""

    def test_screw_to_transform_matches_exponential(self) -> None:
        """Test screw_to_transform matches screw_axis + exponential_map."""
        axis = np.array([0, 0, 1])
        point = np.array([1, 0, 0])
        pitch = 0
        theta = np.pi / 2

        T1 = screw_to_transform(axis, point, pitch, theta)

        S = screw_axis(axis, point, pitch)
        T2 = exponential_map(S, theta)

        np.testing.assert_allclose(T1, T2, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
