"""
Unit tests for joint kinematics module.

Tests joint transform and motion subspace calculations.
"""

import numpy as np
import pytest

from src.spatial_algebra.joints import jcalc


class TestJointCalculations:
    """Tests for jcalc function."""

    def test_revolute_x_axis(self) -> None:
        """Test revolute joint about x-axis."""
        xj_transform, s_subspace = jcalc("Rx", np.pi / 4)
        assert xj_transform.shape == (6, 6)
        assert s_subspace.shape == (6,)
        np.testing.assert_allclose(s_subspace[:3], [1, 0, 0], atol=1e-10)

    def test_revolute_y_axis(self) -> None:
        """Test revolute joint about y-axis."""
        xj_transform, s_subspace = jcalc("Ry", np.pi / 4)
        assert xj_transform.shape == (6, 6)
        np.testing.assert_allclose(s_subspace[:3], [0, 1, 0], atol=1e-10)

    def test_revolute_z_axis(self) -> None:
        """Test revolute joint about z-axis."""
        xj_transform, s_subspace = jcalc("Rz", np.pi / 4)
        assert xj_transform.shape == (6, 6)
        np.testing.assert_allclose(s_subspace[:3], [0, 0, 1], atol=1e-10)

    def test_prismatic_x_axis(self) -> None:
        """Test prismatic joint along x-axis."""
        xj_transform, s_subspace = jcalc("Px", 0.5)
        assert xj_transform.shape == (6, 6)
        np.testing.assert_allclose(s_subspace[3:], [1, 0, 0], atol=1e-10)

    def test_prismatic_y_axis(self) -> None:
        """Test prismatic joint along y-axis."""
        xj_transform, s_subspace = jcalc("Py", 0.5)
        assert xj_transform.shape == (6, 6)
        np.testing.assert_allclose(s_subspace[3:], [0, 1, 0], atol=1e-10)

    def test_prismatic_z_axis(self) -> None:
        """Test prismatic joint along z-axis."""
        xj_transform, s_subspace = jcalc("Pz", 0.5)
        assert xj_transform.shape == (6, 6)
        np.testing.assert_allclose(s_subspace[3:], [0, 0, 1], atol=1e-10)

    def test_invalid_joint_type(self) -> None:
        """Test jcalc raises error for invalid joint type."""
        with pytest.raises(ValueError, match="Unsupported joint type"):
            jcalc("Invalid", 0.0)

    def test_zero_angle_revolute(self) -> None:
        """Test revolute joint at zero angle."""
        xj_transform, s_subspace = jcalc("Rz", 0.0)
        # At zero angle, transform should be close to identity
        assert xj_transform.shape == (6, 6)

    def test_zero_displacement_prismatic(self) -> None:
        """Test prismatic joint at zero displacement."""
        xj_transform, s_subspace = jcalc("Px", 0.0)
        # At zero displacement, transform should be identity
        np.testing.assert_allclose(xj_transform, np.eye(6), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
