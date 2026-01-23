"""
Unit tests for spatial algebra module.

Tests spatial vectors, transformations, and inertia operations.
"""

import numpy as np
import pytest

from src.spatial_algebra import (
    crf,
    crm,
    inv_xtrans,
    mcI,
    spatial_cross,
    transform_spatial_inertia,
    xlt,
    xrot,
    xtrans,
)


class TestSpatialCrossProducts:
    """Tests for spatial cross product operators."""

    def test_crm_shape(self) -> None:
        """Test CRM returns correct shape."""
        v = np.array([1, 0, 0, 0, 1, 0])
        x_matrix = crm(v)
        assert x_matrix.shape == (6, 6)

    def test_crf_crm_relationship(self) -> None:
        """Test that crf(v) = -crm(v).T"""
        v = np.array([1, 2, 3, 4, 5, 6])
        x_crf = crf(v)
        x_crm = crm(v)
        np.testing.assert_allclose(x_crf, -x_crm.T, atol=1e-10)

    def test_spatial_cross_motion(self) -> None:
        """Test motion cross product."""
        v1 = np.array([1, 0, 0, 0, 0, 0])  # Pure angular about x
        v2 = np.array([0, 1, 0, 0, 0, 0])  # Pure angular about y
        result = spatial_cross(v1, v2, "motion")
        expected = np.array([0, 0, 1, 0, 0, 0])  # Should be about z
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_spatial_cross_force(self) -> None:
        """Test force cross product."""
        v = np.array([1, 0, 0, 0, 1, 0])
        f = np.array([0, 0, 10, 0, 0, 0])
        result = spatial_cross(v, f, "force")
        assert result.shape == (6,)
        # Verify using crf operator directly
        expected = crf(v) @ f
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_spatial_cross_invalid_type(self) -> None:
        """Test spatial_cross raises error for invalid type."""
        v = np.array([1, 0, 0, 0, 0, 0])
        u = np.array([0, 1, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="cross_type must be"):
            spatial_cross(v, u, "invalid")

    def test_crm_zero_velocity_returns_zero_matrix(self) -> None:
        """Test CRM returns zero matrix for zero velocity."""
        v = np.zeros(6)
        x_matrix = crm(v)
        np.testing.assert_allclose(x_matrix, np.zeros((6, 6)), atol=1e-10)

    def test_crf_zero_velocity_returns_zero_matrix(self) -> None:
        """Test CRF returns zero matrix for zero velocity."""
        v = np.zeros(6)
        x_matrix = crf(v)
        np.testing.assert_allclose(x_matrix, np.zeros((6, 6)), atol=1e-10)


class TestSpatialTransforms:
    """Tests for spatial transformation matrices."""

    def test_xrot_identity(self) -> None:
        """Test xrot with identity rotation."""
        e_rot = np.eye(3)
        x_matrix = xrot(e_rot)
        assert x_matrix.shape == (6, 6)
        np.testing.assert_allclose(x_matrix[:3, :3], e_rot, atol=1e-10)
        np.testing.assert_allclose(x_matrix[3:, 3:], e_rot, atol=1e-10)

    def test_xrot_90_degree_z(self) -> None:
        """Test xrot with 90 degree rotation about z-axis."""
        theta = np.pi / 2
        e_rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        x_matrix = xrot(e_rot)
        assert x_matrix.shape == (6, 6)

    def test_xlt_translation(self) -> None:
        """Test xlt with translation vector."""
        r = np.array([1, 2, 3])
        x_matrix = xlt(r)
        assert x_matrix.shape == (6, 6)
        # Upper left should be identity
        np.testing.assert_allclose(x_matrix[:3, :3], np.eye(3), atol=1e-10)
        # Lower right should be identity
        np.testing.assert_allclose(x_matrix[3:, 3:], np.eye(3), atol=1e-10)

    def test_xlt_zero_translation(self) -> None:
        """Test xlt with zero translation."""
        r = np.zeros(3)
        x_matrix = xlt(r)
        np.testing.assert_allclose(x_matrix, np.eye(6), atol=1e-10)

    def test_xtrans_identity(self) -> None:
        """Test xtrans with identity rotation and zero translation."""
        e_rot = np.eye(3)
        r = np.zeros(3)
        x_matrix = xtrans(e_rot, r)
        assert x_matrix.shape == (6, 6)
        np.testing.assert_allclose(x_matrix, np.eye(6), atol=1e-10)

    def test_xtrans_rotation_and_translation(self) -> None:
        """Test xtrans with rotation and translation."""
        e_rot = np.eye(3)
        r = np.array([1, 0, 0])
        x_matrix = xtrans(e_rot, r)
        assert x_matrix.shape == (6, 6)

    def test_inv_xtrans_inverse_property(self) -> None:
        """Test inv_xtrans is inverse of xtrans."""
        e_rot = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        r = np.array([1, 2, 3])
        x_matrix = xtrans(e_rot, r)
        x_inv = inv_xtrans(e_rot, r)
        # X @ X_inv should be identity
        result = x_matrix @ x_inv
        np.testing.assert_allclose(result, np.eye(6), atol=1e-10)

    def test_xrot_invalid_rotation_matrix(self) -> None:
        """Test xrot raises error for invalid rotation matrix."""
        e_rot = np.eye(3) * 2
        with pytest.raises(ValueError, match="rotation matrix"):
            xrot(e_rot)


class TestSpatialInertia:
    """Tests for spatial inertia matrices."""

    def test_mci_shape(self) -> None:
        """Test mcI returns correct shape."""
        mass = 1.0
        com = np.array([0, 0, 0])
        i_com = np.eye(3) * 0.1
        i_spatial = mcI(mass, com, i_com)
        assert i_spatial.shape == (6, 6)

    def test_mci_symmetry(self) -> None:
        """Test spatial inertia matrix is symmetric."""
        mass = 1.0
        com = np.array([0.1, 0.2, 0.3])
        i_com = np.eye(3) * 0.1
        i_spatial = mcI(mass, com, i_com)
        np.testing.assert_allclose(i_spatial, i_spatial.T, atol=1e-10)

    def test_mci_zero_com(self) -> None:
        """Test mcI with zero COM."""
        mass = 1.0
        com = np.zeros(3)
        i_com = np.eye(3) * 0.1
        i_spatial = mcI(mass, com, i_com)
        # Should have block diagonal structure
        assert i_spatial.shape == (6, 6)

    def test_mci_positive_mass(self) -> None:
        """Test mcI requires positive mass."""
        with pytest.raises(ValueError, match="mass must be positive"):
            mcI(0.0, np.zeros(3), np.eye(3))

    def test_transform_spatial_inertia_identity(self) -> None:
        """Test transform_spatial_inertia with identity transform."""
        i_b = mcI(1.0, np.zeros(3), np.eye(3) * 0.1)
        x_matrix = np.eye(6)
        i_a = transform_spatial_inertia(i_b, x_matrix)
        np.testing.assert_allclose(i_a, i_b, atol=1e-10)

    def test_transform_spatial_inertia_symmetry(self) -> None:
        """Test transformed inertia remains symmetric."""
        i_b = mcI(1.0, np.array([0.1, 0.2, 0.3]), np.eye(3) * 0.1)
        e_rot = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        r = np.array([1, 0, 0])
        x_matrix = xtrans(e_rot, r)
        i_a = transform_spatial_inertia(i_b, x_matrix)
        np.testing.assert_allclose(i_a, i_a.T, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
