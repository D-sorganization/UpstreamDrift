"""
Unit tests for spatial algebra module.

Tests spatial vectors, transformations, and inertia operations.
"""

import numpy as np
import pytest
from mujoco_humanoid_golf.spatial_algebra import (
    crf,
    crm,
    inv_xtrans,
    jcalc,
    mci,
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
        X = crm(v)
        assert X.shape == (6, 6)

    def test_crf_crm_relationship(self) -> None:
        """Test that crf(v) = -crm(v).T"""
        v = np.array([1, 2, 3, 4, 5, 6])
        X_crf = crf(v)
        X_crm = crm(v)
        np.testing.assert_allclose(X_crf, -X_crm.T, atol=1e-10)

    def test_spatial_cross_motion(self) -> None:
        """Test motion cross product."""
        v1 = np.array([1, 0, 0, 0, 0, 0])  # Pure angular about x
        v2 = np.array([0, 1, 0, 0, 0, 0])  # Pure angular about y
        result = spatial_cross(v1, v2, "motion")
        expected = np.array([0, 0, 1, 0, 0, 0])  # Should be about z
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_spatial_cross_force(self) -> None:
        """Test force cross product."""
        # v = [1, 0, 0, 0, 1, 0] = [omega_x=1, omega_y=0, omega_z=0,
        #                           v_x=0, v_y=1, v_z=0]
        # f = [0, 0, 10, 0, 0, 0] = [moment_x=0, moment_y=0, moment_z=10,
        #                            force_x=0, force_y=0, force_z=0]
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
        with pytest.raises(ValueError):
            spatial_cross(v, u, "invalid")

    def test_crm_zero_velocity_returns_zero_matrix(self) -> None:
        """Test CRM returns zero matrix for zero velocity."""
        v = np.zeros(6)
        X = crm(v)
        np.testing.assert_allclose(X, np.zeros((6, 6)), atol=1e-10)

    def test_crf_zero_velocity_returns_zero_matrix(self) -> None:
        """Test CRF returns zero matrix for zero velocity."""
        v = np.zeros(6)
        X = crf(v)
        np.testing.assert_allclose(X, np.zeros((6, 6)), atol=1e-10)


class TestSpatialTransforms:
    """Tests for spatial transformations."""

    def test_xrot_block_diagonal(self) -> None:
        """Test XROT has block diagonal structure."""
        theta = np.pi / 2
        E = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
        )
        X = xrot(E)

        # Check off-diagonal blocks are zero
        np.testing.assert_allclose(X[:3, 3:], 0, atol=1e-10)
        np.testing.assert_allclose(X[3:, :3], 0, atol=1e-10)

        # Check diagonal blocks equal E
        np.testing.assert_allclose(X[:3, :3], E, atol=1e-10)
        np.testing.assert_allclose(X[3:, 3:], E, atol=1e-10)

    def test_xlt_structure(self) -> None:
        """Test XLT has correct structure."""
        r = np.array([1, 2, 3])
        X = xlt(r)

        # Top-right should be zero
        np.testing.assert_allclose(X[:3, 3:], 0, atol=1e-10)

        # Diagonal blocks should be identity
        np.testing.assert_allclose(X[:3, :3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(X[3:, 3:], np.eye(3), atol=1e-10)

    def test_xtrans_inverse(self) -> None:
        """Test XTRANS and INV_XTRANS are inverses."""
        E = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° about z
        r = np.array([1, 2, 0])

        X = xtrans(E, r)
        X_inv = inv_xtrans(E, r)

        product = X @ X_inv
        np.testing.assert_allclose(product, np.eye(6), atol=1e-10)

    def test_xtrans_identity(self) -> None:
        """Test XTRANS with identity rotation."""
        E = np.eye(3)
        r = np.array([1, 2, 3])
        X = xtrans(E, r)

        # Should match xlt
        X_expected = xlt(r)
        np.testing.assert_allclose(X, X_expected, atol=1e-10)

    def test_xrot_invalid_matrix(self) -> None:
        """Test XROT raises error for non-rotation matrix."""
        E = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Not a rotation
        with pytest.raises(ValueError):
            xrot(E)

    def test_xlt_zero_translation(self) -> None:
        """Test XLT with zero translation."""
        r = np.zeros(3)
        X = xlt(r)
        np.testing.assert_allclose(X, np.eye(6), atol=1e-10)


class TestSpatialInertia:
    """Tests for spatial inertia operations."""

    def test_mci_symmetry(self) -> None:
        """Test spatial inertia is symmetric."""
        mass = 1.0
        radius = 0.1
        com = np.array([0, 0, 0])
        I_sphere = (2 / 5) * mass * radius**2 * np.eye(3)

        I_spatial = mci(mass, com, I_sphere)

        # Check symmetry
        np.testing.assert_allclose(I_spatial, I_spatial.T, atol=1e-10)

    def test_mci_positive_definite(self) -> None:
        """Test spatial inertia is positive definite."""
        mass = 2.0
        com = np.array([0.1, 0, 0])
        I_com = 0.01 * np.eye(3)

        I_spatial = mci(mass, com, I_com)

        # All eigenvalues should be positive
        eigvals = np.linalg.eigvals(I_spatial)
        assert np.all(eigvals > 0)

    def test_transform_spatial_inertia_identity(self) -> None:
        """Test identity transform preserves inertia."""
        I_B = mci(1.0, np.zeros(3), 0.01 * np.eye(3))
        X = np.eye(6)
        I_A = transform_spatial_inertia(I_B, X)

        np.testing.assert_allclose(I_A, I_B, atol=1e-10)

    def test_mci_with_offset_com(self) -> None:
        """Test spatial inertia with offset center of mass."""
        mass = 1.0
        com = np.array([0.5, 0, 0])  # COM offset
        I_com = 0.01 * np.eye(3)
        I_spatial = mci(mass, com, I_com)

        # Should still be symmetric and positive definite
        np.testing.assert_allclose(I_spatial, I_spatial.T, atol=1e-10)
        eigvals = np.linalg.eigvals(I_spatial)
        assert np.all(eigvals > 0)

    def test_transform_spatial_inertia_rotation(self) -> None:
        """Test transforming inertia with rotation."""
        I_B = mci(1.0, np.zeros(3), 0.01 * np.eye(3))
        # 90 degree rotation about z
        theta = np.pi / 2
        E = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
        )
        X = xrot(E)
        I_A = transform_spatial_inertia(I_B, X)

        # Should still be symmetric and positive definite
        np.testing.assert_allclose(I_A, I_A.T, atol=1e-10)
        eigvals = np.linalg.eigvals(I_A)
        assert np.all(eigvals > 0)


class TestJointCalculations:
    """Tests for joint kinematics."""

    def test_jcalc_revolute_z(self) -> None:
        """Test revolute joint about z-axis."""
        Xj, S = jcalc("Rz", np.pi / 4)

        # Motion subspace should be [0, 0, 1, 0, 0, 0]
        expected_S = np.array([0, 0, 1, 0, 0, 0])
        np.testing.assert_allclose(S, expected_S, atol=1e-10)

        # Xj should be 6x6
        assert Xj.shape == (6, 6)

    def test_jcalc_prismatic_x(self) -> None:
        """Test prismatic joint along x-axis."""
        _Xj, S = jcalc("Px", 0.5)

        # Motion subspace should be [0, 0, 0, 1, 0, 0]
        expected_S = np.array([0, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(S, expected_S, atol=1e-10)

    def test_jcalc_all_joint_types(self) -> None:
        """Test all supported joint types."""
        joint_types = ["Rx", "Ry", "Rz", "Px", "Py", "Pz"]

        for jtype in joint_types:
            Xj, S = jcalc(jtype, 0.1)
            assert Xj.shape == (6, 6)
            assert S.shape == (6,)

    def test_jcalc_unsupported_joint(self) -> None:
        """Test error for unsupported joint type."""
        with pytest.raises(ValueError):
            jcalc("invalid", 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
