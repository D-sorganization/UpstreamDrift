"""Tests for motion retargeting module."""

from __future__ import annotations

import numpy as np
import pytest


class TestSkeletonConfig:
    """Tests for SkeletonConfig."""

    def test_skeleton_creation(self) -> None:
        """Test creating a skeleton configuration."""
        from src.learning.retargeting import SkeletonConfig

        skeleton = SkeletonConfig(
            name="test",
            joint_names=["root", "joint1", "joint2"],
            parent_indices=[-1, 0, 1],
            joint_offsets=np.array([
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]),
        )

        assert skeleton.n_joints == 3
        assert skeleton.name == "test"

    def test_skeleton_get_joint_index(self) -> None:
        """Test getting joint index by name."""
        from src.learning.retargeting import SkeletonConfig

        skeleton = SkeletonConfig(
            name="test",
            joint_names=["root", "joint1", "joint2"],
            parent_indices=[-1, 0, 1],
            joint_offsets=np.zeros((3, 3)),
        )

        assert skeleton.get_joint_index("root") == 0
        assert skeleton.get_joint_index("joint1") == 1
        assert skeleton.get_joint_index("joint2") == 2

        with pytest.raises(ValueError):
            skeleton.get_joint_index("nonexistent")

    def test_skeleton_semantic_labels(self) -> None:
        """Test semantic label mapping."""
        from src.learning.retargeting import SkeletonConfig

        skeleton = SkeletonConfig(
            name="test",
            joint_names=["j0", "j1", "j2"],
            parent_indices=[-1, 0, 1],
            joint_offsets=np.zeros((3, 3)),
            semantic_labels={"left_shoulder": "j1"},
        )

        assert skeleton.get_semantic_joint("left_shoulder") == "j1"
        assert skeleton.get_semantic_joint("nonexistent") is None

    def test_skeleton_kinematic_chain(self) -> None:
        """Test getting kinematic chain."""
        from src.learning.retargeting import SkeletonConfig

        skeleton = SkeletonConfig(
            name="test",
            joint_names=["root", "spine", "chest", "shoulder", "elbow"],
            parent_indices=[-1, 0, 1, 2, 3],
            joint_offsets=np.zeros((5, 3)),
        )

        chain = skeleton.get_kinematic_chain("elbow")
        assert chain == ["root", "spine", "chest", "shoulder", "elbow"]

    def test_create_humanoid(self) -> None:
        """Test creating standard humanoid skeleton."""
        from src.learning.retargeting import SkeletonConfig

        humanoid = SkeletonConfig.create_humanoid()

        assert humanoid.n_joints == 22
        assert "pelvis" in humanoid.joint_names
        assert "left_hand" in humanoid.joint_names
        assert len(humanoid.end_effectors) == 5


class TestMotionRetargeter:
    """Tests for MotionRetargeter."""

    def create_simple_skeleton(
        self, name: str, n_joints: int = 5
    ) -> "SkeletonConfig":
        """Create a simple test skeleton."""
        from src.learning.retargeting import SkeletonConfig

        return SkeletonConfig(
            name=name,
            joint_names=[f"joint_{i}" for i in range(n_joints)],
            parent_indices=[-1] + list(range(n_joints - 1)),
            joint_offsets=np.array([[0, 0, i * 0.1] for i in range(n_joints)]),
            semantic_labels={
                "root": "joint_0",
                "spine": "joint_1",
                "chest": "joint_2",
                "shoulder": "joint_3",
                "hand": "joint_4",
            },
            end_effectors=["joint_4"],
        )

    def test_retargeter_creation(self) -> None:
        """Test creating a motion retargeter."""
        from src.learning.retargeting import MotionRetargeter

        source = self.create_simple_skeleton("source")
        target = self.create_simple_skeleton("target")

        retargeter = MotionRetargeter(source, target)

        assert retargeter.source is source
        assert retargeter.target is target

    def test_joint_mapping(self) -> None:
        """Test automatic joint mapping."""
        from src.learning.retargeting import MotionRetargeter

        source = self.create_simple_skeleton("source")
        target = self.create_simple_skeleton("target")

        retargeter = MotionRetargeter(source, target)
        mapping = retargeter.get_joint_mapping()

        # Should map all joints with matching semantic labels
        assert len(mapping) == 5

    def test_direct_retargeting(self) -> None:
        """Test direct joint angle retargeting."""
        from src.learning.retargeting import MotionRetargeter

        source = self.create_simple_skeleton("source")
        target = self.create_simple_skeleton("target")

        retargeter = MotionRetargeter(source, target)

        # Create source motion
        n_frames = 20
        source_motion = np.random.randn(n_frames, source.n_joints)

        # Retarget
        target_motion = retargeter.retarget(source_motion, method="direct")

        assert target_motion.shape == (n_frames, target.n_joints)

    def test_retargeting_preserves_angles(self) -> None:
        """Test that direct retargeting preserves angles for matching skeletons."""
        from src.learning.retargeting import MotionRetargeter

        skeleton = self.create_simple_skeleton("skeleton")

        retargeter = MotionRetargeter(skeleton, skeleton)

        source_motion = np.random.randn(10, skeleton.n_joints)
        target_motion = retargeter.retarget(source_motion, method="direct")

        np.testing.assert_array_almost_equal(source_motion, target_motion)

    def test_humanoid_retargeting(self) -> None:
        """Test retargeting between humanoid skeletons."""
        from src.learning.retargeting import MotionRetargeter, SkeletonConfig

        source = SkeletonConfig.create_humanoid()
        target = SkeletonConfig.create_humanoid()

        retargeter = MotionRetargeter(source, target)

        # Create motion
        n_frames = 50
        source_motion = np.random.randn(n_frames, source.n_joints) * 0.5

        target_motion = retargeter.retarget(source_motion, method="direct")

        assert target_motion.shape == (n_frames, target.n_joints)

    def test_visualize_mapping(self) -> None:
        """Test mapping visualization."""
        from src.learning.retargeting import MotionRetargeter

        source = self.create_simple_skeleton("source")
        target = self.create_simple_skeleton("target")

        retargeter = MotionRetargeter(source, target)
        viz = retargeter.visualize_mapping()

        assert "source" in viz
        assert "target" in viz
        assert "Joint Mapping" in viz

    def test_mocap_retargeting(self) -> None:
        """Test motion capture retargeting."""
        from src.learning.retargeting import MotionRetargeter

        source = self.create_simple_skeleton("source")
        target = self.create_simple_skeleton("target")

        retargeter = MotionRetargeter(source, target)

        # Create mock mocap data
        n_frames = 30
        n_markers = 5
        marker_positions = np.random.randn(n_frames, n_markers, 3)
        marker_names = ["LSHO", "RSHO", "LELB", "RELB", "LWRI"]

        target_motion = retargeter.retarget_from_mocap(
            marker_positions, marker_names
        )

        assert target_motion.shape == (n_frames, target.n_joints)
