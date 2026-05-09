"""Tests for pose_editor.core and pose_editor.library (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.pose_editor.core import JointInfo, JointType, PoseEditorState
from src.shared.python.pose_editor.library import (
    PoseLibrary,
    StoredPose,
    get_preset_pose,
    list_preset_poses,
    list_preset_poses_by_category,
)


class TestJointType:
    def test_revolute_exists(self) -> None:
        assert JointType.REVOLUTE is not None

    def test_prismatic_exists(self) -> None:
        assert JointType.PRISMATIC is not None


class TestPoseEditorState:
    def test_pose_editor_construction(self) -> None:
        state = PoseEditorState()
        assert state is not None


class TestJointInfo:
    def test_pose_editor_construction(self) -> None:
        ji = JointInfo(
            name="shoulder",
            index=0,
            joint_type=JointType.REVOLUTE,
            position_index=0,
            velocity_index=0,
            num_positions=1,
            num_velocities=1,
        )
        assert ji.name == "shoulder"

    def test_joint_type(self) -> None:
        ji = JointInfo(
            name="knee",
            index=1,
            joint_type=JointType.REVOLUTE,
            position_index=1,
            velocity_index=1,
            num_positions=1,
            num_velocities=1,
        )
        assert ji.joint_type == JointType.REVOLUTE


class TestListPresetPoses:
    def test_pose_editor_returns_list(self) -> None:
        poses = list_preset_poses()
        assert isinstance(poses, list)

    def test_pose_editor_non_empty(self) -> None:
        poses = list_preset_poses()
        assert len(poses) > 0

    def test_contains_address(self) -> None:
        poses = list_preset_poses()
        assert "Address" in poses


class TestGetPresetPose:
    def test_returns_dict_or_none(self) -> None:
        result = get_preset_pose("Address")
        assert result is None or isinstance(result, dict)

    def test_unknown_pose_returns_none(self) -> None:
        result = get_preset_pose("NonExistentPose12345")
        assert result is None


class TestListPresetPosesByCategory:
    def test_pose_editor_returns_list(self) -> None:
        poses = list_preset_poses_by_category("golf")
        assert isinstance(poses, list)


class TestPoseLibrary:
    def test_pose_editor_construction(self) -> None:
        lib = PoseLibrary()
        assert lib is not None

    def test_list_poses_returns_list(self) -> None:
        lib = PoseLibrary()
        poses = lib.list_poses()
        assert isinstance(poses, list)

    def test_load_nonexistent_returns_none(self) -> None:
        lib = PoseLibrary()
        result = lib.load_pose("NonExistentXYZ999")
        assert result is None


class TestStoredPose:
    def test_pose_editor_construction(self) -> None:
        pose = StoredPose(
            name="test_pose",
            joint_positions=np.zeros(5),
        )
        assert pose.name == "test_pose"

    def test_joint_positions(self) -> None:
        positions = np.array([0.1, 0.2, 0.3])
        pose = StoredPose(name="test", joint_positions=positions)
        np.testing.assert_allclose(pose.joint_positions, positions)
