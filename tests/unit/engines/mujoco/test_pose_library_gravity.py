"""
Integration tests for pose library with gravity interpolation.

Tests that pose interpolation correctly handles gravity vectors for
non-linear paths through configuration space.

Related to issue #4106: Gravity vector interpolation fails for non-linear
parametric paths through config space.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._pose_library import (
    PoseLibraryMixin,
    StoredPose,
)


class MockMuJoCoModel(PoseLibraryMixin):
    """Mock MuJoCo simulation for testing pose library."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(
            """
<mujoco model="test">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="test" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="1 0 0"/>
      <joint name="joint2" type="hinge" axis="0 1 0"/>
      <geom name="geom1" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
            """
        )
        self.data = mujoco.MjData(self.model)
        self.pose_library: dict[str, StoredPose] = {}


# =============================================================================
# Test Suite: Pose Library with Gravity Interpolation
# =============================================================================


class TestPoseLibraryGravityBasics:
    """Test basic gravity handling in pose library."""

    def test_stored_pose_with_gravity(self) -> None:
        """StoredPose should store gravity vectors."""
        pose = StoredPose(
            name="test_pose",
            qpos=np.array([0.1, 0.2]),
            gravity=np.array([0.0, 0.0, -9.81]),
        )

        assert pose.gravity is not None
        np.testing.assert_array_equal(pose.gravity, np.array([0.0, 0.0, -9.81]))

    def test_stored_pose_without_gravity(self) -> None:
        """StoredPose should work without gravity vectors (backward compat)."""
        pose = StoredPose(
            name="test_pose",
            qpos=np.array([0.1, 0.2]),
        )

        assert pose.gravity is None

    def test_pose_library_export_import_with_gravity(self) -> None:
        """Pose library should preserve gravity in export/import."""
        model = MockMuJoCoModel()

        # Create and save poses with gravity
        pose1 = StoredPose(
            name="pose1",
            qpos=np.array([0.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),
        )
        pose2 = StoredPose(
            name="pose2",
            qpos=np.array([0.5, 0.5]),
            gravity=np.array([9.81, 0.0, 0.0]),
        )

        model.pose_library["pose1"] = pose1
        model.pose_library["pose2"] = pose2

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            model.export_pose_library(temp_path)

            # Clear and reimport
            model.pose_library.clear()
            count = model.import_pose_library(temp_path)

            assert count == 2
            assert "pose1" in model.pose_library
            assert "pose2" in model.pose_library

            # Check gravity was preserved
            imported_pose1 = model.pose_library["pose1"]
            imported_pose2 = model.pose_library["pose2"]

            assert imported_pose1.gravity is not None
            np.testing.assert_array_equal(
                imported_pose1.gravity, np.array([0.0, 0.0, -9.81])
            )

            assert imported_pose2.gravity is not None
            np.testing.assert_array_equal(
                imported_pose2.gravity, np.array([9.81, 0.0, 0.0])
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPoseLibraryGravityInterpolation:
    """Test gravity interpolation in pose library."""

    def test_interpolate_poses_without_gravity(self) -> None:
        """Interpolation should work without gravity (backward compat)."""
        model = MockMuJoCoModel()

        pose_a = StoredPose(
            name="pose_a",
            qpos=np.array([0.0, 0.0]),
        )
        pose_b = StoredPose(
            name="pose_b",
            qpos=np.array([1.0, 1.0]),
        )

        model.pose_library["pose_a"] = pose_a
        model.pose_library["pose_b"] = pose_b

        # Should succeed and interpolate qpos
        success = model.interpolate_poses("pose_a", "pose_b", 0.5)
        assert success

        # Check qpos was interpolated
        expected_qpos = np.array([0.5, 0.5])
        np.testing.assert_allclose(model.data.qpos[:2], expected_qpos, atol=1e-10)

        # Gravity should not have changed from default
        np.testing.assert_allclose(
            model.model.opt.gravity, np.array([0.0, 0.0, -9.81]), atol=1e-10
        )

    def test_interpolate_poses_with_gravity_vertical(self) -> None:
        """Interpolation with same gravity direction."""
        model = MockMuJoCoModel()

        pose_a = StoredPose(
            name="pose_a",
            qpos=np.array([0.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),
        )
        pose_b = StoredPose(
            name="pose_b",
            qpos=np.array([1.0, 1.0]),
            gravity=np.array([0.0, 0.0, -9.81]),
        )

        model.pose_library["pose_a"] = pose_a
        model.pose_library["pose_b"] = pose_b

        success = model.interpolate_poses("pose_a", "pose_b", 0.5)
        assert success

        # qpos should be interpolated
        expected_qpos = np.array([0.5, 0.5])
        np.testing.assert_allclose(model.data.qpos[:2], expected_qpos, atol=1e-10)

        # Gravity should still be vertical
        np.testing.assert_allclose(
            model.model.opt.gravity,
            np.array([0.0, 0.0, -9.81]),
            atol=1e-10,
        )

    def test_interpolate_poses_with_gravity_rotation(self) -> None:
        """Interpolation with different gravity directions."""
        model = MockMuJoCoModel()

        # Down to right rotation (90 degrees)
        pose_a = StoredPose(
            name="pose_a",
            qpos=np.array([0.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),
        )
        pose_b = StoredPose(
            name="pose_b",
            qpos=np.array([1.0, 1.0]),
            gravity=np.array([9.81, 0.0, 0.0]),
        )

        model.pose_library["pose_a"] = pose_a
        model.pose_library["pose_b"] = pose_b

        # Interpolate at midpoint
        success = model.interpolate_poses("pose_a", "pose_b", 0.5)
        assert success

        # qpos should be interpolated
        expected_qpos = np.array([0.5, 0.5])
        np.testing.assert_allclose(model.data.qpos[:2], expected_qpos, atol=1e-10)

        # Gravity should be interpolated
        g_interp = model.model.opt.gravity
        mag = np.linalg.norm(g_interp)

        # Magnitude should be 9.81
        assert np.isclose(mag, 9.81, atol=1e-10)

        # Direction should be roughly between down and right
        # (both x and z components should be significant)
        assert np.abs(g_interp[0]) > 1.0  # x component should be nonzero
        assert g_interp[2] < -1.0  # z component should be significantly negative

    def test_interpolate_poses_at_endpoints(self) -> None:
        """Interpolation at endpoints should match exactly."""
        model = MockMuJoCoModel()

        pose_a = StoredPose(
            name="pose_a",
            qpos=np.array([0.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),
        )
        pose_b = StoredPose(
            name="pose_b",
            qpos=np.array([1.0, 1.0]),
            gravity=np.array([9.81, 0.0, 0.0]),
        )

        model.pose_library["pose_a"] = pose_a
        model.pose_library["pose_b"] = pose_b

        # At alpha=0
        model.interpolate_poses("pose_a", "pose_b", 0.0)
        np.testing.assert_allclose(model.data.qpos[:2], pose_a.qpos, atol=1e-10)
        np.testing.assert_allclose(
            model.model.opt.gravity, pose_a.gravity, atol=1e-10
        )

        # At alpha=1
        model.interpolate_poses("pose_a", "pose_b", 1.0)
        np.testing.assert_allclose(model.data.qpos[:2], pose_b.qpos, atol=1e-10)
        np.testing.assert_allclose(
            model.model.opt.gravity, pose_b.gravity, atol=1e-10
        )

    def test_interpolate_poses_with_varying_magnitudes(self) -> None:
        """Interpolation should handle different gravity magnitudes."""
        model = MockMuJoCoModel()

        pose_a = StoredPose(
            name="pose_a",
            qpos=np.array([0.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),  # Earth
        )
        pose_b = StoredPose(
            name="pose_b",
            qpos=np.array([1.0, 1.0]),
            gravity=np.array([0.0, 0.0, -1.6]),  # Moon
        )

        model.pose_library["pose_a"] = pose_a
        model.pose_library["pose_b"] = pose_b

        success = model.interpolate_poses("pose_a", "pose_b", 0.5)
        assert success

        g_interp = model.model.opt.gravity
        mag = np.linalg.norm(g_interp)

        # Magnitude should be interpolated
        expected_mag = 0.5 * 9.81 + 0.5 * 1.6
        assert np.isclose(mag, expected_mag, atol=1e-10)

    def test_interpolate_poses_missing_pose_returns_false(self) -> None:
        """Interpolation should return False if pose not found."""
        model = MockMuJoCoModel()

        pose_a = StoredPose(
            name="pose_a",
            qpos=np.array([0.0, 0.0]),
        )

        model.pose_library["pose_a"] = pose_a

        # Try to interpolate with non-existent pose
        success = model.interpolate_poses("pose_a", "non_existent", 0.5)
        assert not success

    def test_interpolate_poses_alpha_clamping(self) -> None:
        """Interpolation should clamp alpha to [0, 1]."""
        model = MockMuJoCoModel()

        pose_a = StoredPose(
            name="pose_a",
            qpos=np.array([0.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),
        )
        pose_b = StoredPose(
            name="pose_b",
            qpos=np.array([1.0, 1.0]),
            gravity=np.array([9.81, 0.0, 0.0]),
        )

        model.pose_library["pose_a"] = pose_a
        model.pose_library["pose_b"] = pose_b

        # Alpha > 1 should clamp to 1
        model.interpolate_poses("pose_a", "pose_b", 2.0)
        np.testing.assert_allclose(model.data.qpos[:2], pose_b.qpos, atol=1e-10)

        # Alpha < 0 should clamp to 0
        model.interpolate_poses("pose_a", "pose_b", -1.0)
        np.testing.assert_allclose(model.data.qpos[:2], pose_a.qpos, atol=1e-10)


class TestPoseLibraryGravitySequence:
    """Test gravity handling in pose sequences."""

    def test_interpolate_sequence_three_non_collinear_gravity_directions(
        self,
    ) -> None:
        """Test interpolation along a path with three non-collinear gravity directions."""
        model = MockMuJoCoModel()

        # Create three poses with non-collinear gravity directions
        pose1 = StoredPose(
            name="pose1",
            qpos=np.array([0.0, 0.0]),
            gravity=np.array([0.0, 0.0, -9.81]),  # Down
        )
        pose2 = StoredPose(
            name="pose2",
            qpos=np.array([0.5, 0.5]),
            gravity=np.array([9.81, 0.0, 0.0]),  # Right
        )
        pose3 = StoredPose(
            name="pose3",
            qpos=np.array([1.0, 1.0]),
            gravity=np.array([0.0, 9.81, 0.0]),  # Forward
        )

        model.pose_library["pose1"] = pose1
        model.pose_library["pose2"] = pose2
        model.pose_library["pose3"] = pose3

        # Interpolate along segment 1->2
        for alpha in np.linspace(0, 1, 5):
            success = model.interpolate_poses("pose1", "pose2", alpha)
            assert success
            g = model.model.opt.gravity
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 9.81, atol=1e-10)

        # Interpolate along segment 2->3
        for alpha in np.linspace(0, 1, 5):
            success = model.interpolate_poses("pose2", "pose3", alpha)
            assert success
            g = model.model.opt.gravity
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 9.81, atol=1e-10)

        # Interpolate along segment 3->1
        for alpha in np.linspace(0, 1, 5):
            success = model.interpolate_poses("pose3", "pose1", alpha)
            assert success
            g = model.model.opt.gravity
            mag = np.linalg.norm(g)
            assert np.isclose(mag, 9.81, atol=1e-10)
