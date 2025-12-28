"""
Unit tests for spatial algebra module.
"""

import numpy as np
import pytest

from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.spatial_algebra import (
    inertia,
    joints,
    spatial_vectors,
    transforms,
)


class TestSpatialVectors:
    """Test spatial vector operations."""

    def test_spatial_cross_motion(self):
        """Test motion cross product (v x m)."""
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        res = spatial_vectors.cross_motion(v1, v2)
        assert res.shape == (6,)

        # Test property: v x v = 0
        res_self = spatial_vectors.cross_motion(v1, v1)
        np.testing.assert_allclose(res_self, np.zeros(6), atol=1e-10)

    def test_spatial_cross_force(self):
        """Test force cross product (v x* f)."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        res = spatial_vectors.cross_force(v, f)
        assert res.shape == (6,)

    def test_mci(self):
        """Test mci (spatial inertia matrix construction)."""
        mass = 1.0
        center_of_mass = np.array([0.1, 0.2, 0.3])
        inertia_tensor = np.eye(3)

        # Build spatial inertia matrix
        # Note: Function name is mci (lowercase) based on file read
        I_spatial = inertia.mci(mass, center_of_mass, inertia_tensor)
        assert I_spatial.shape == (6, 6)

        # Check symmetry
        np.testing.assert_allclose(I_spatial, I_spatial.T)


class TestTransforms:
    """Test coordinate transforms."""

    def test_xrot(self):
        """Test X-axis rotation."""
        theta = np.pi / 2
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

        X = transforms.xrot(Rx)
        expected = np.zeros((6, 6))
        expected[:3, :3] = Rx
        expected[3:, 3:] = Rx

        np.testing.assert_allclose(X, expected, atol=1e-10)

    def test_xlt(self):
        """Test spatial transform generation from translation."""
        r = np.array([1.0, 2.0, 3.0])

        X = transforms.xlt(r)
        assert X.shape == (6, 6)

        # Identity transform
        X_id = transforms.xlt(np.zeros(3))
        np.testing.assert_allclose(X_id, np.eye(6))

    def test_xtrans(self):
        """Test general spatial transform."""
        theta = np.pi / 2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        r = np.array([1.0, 0.0, 0.0])

        X = transforms.xtrans(R, r)
        assert X.shape == (6, 6)

    def test_inv_xtrans(self):
        """Test inverse transform."""
        theta = np.pi / 4
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        r = np.array([1.0, 2.0, 3.0])

        X = transforms.xtrans(R, r)
        X_inv = transforms.inv_xtrans(R, r)

        # Check X * X_inv = I
        np.testing.assert_allclose(X @ X_inv, np.eye(6), atol=1e-10)


class TestJoints:
    """Test joint subspace generation."""

    def test_jcalc_revolute(self):
        """Test revolute joint subspace."""
        X, S = joints.jcalc("Rx", 0.0)
        assert X.shape == (6, 6)
        assert S.shape == (6,)
        np.testing.assert_allclose(S, np.array([1, 0, 0, 0, 0, 0]))

    def test_jcalc_prismatic(self):
        """Test prismatic joint subspace."""
        X, S = joints.jcalc("Px", 0.0)
        np.testing.assert_allclose(S, np.array([0, 0, 0, 1, 0, 0]))

    def test_unknown_joint(self):
        """Test error handling for unknown joint."""
        with pytest.raises(ValueError):
            joints.jcalc("InvalidType", 0.0)
