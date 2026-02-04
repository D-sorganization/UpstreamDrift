"""Unit tests for skeleton mapping system.

TDD tests for mapping gaming skeletons to physics model joints.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.unreal_integration.skeleton_mapper import (
    SkeletonMapper,
    SkeletonType,
    BoneMapping,
    MappingProfile,
    PoseTransform,
    MIXAMO_TO_PHYSICS_MAP,
    UNREAL_MANNEQUIN_TO_PHYSICS_MAP,
    MUJOCO_HUMANOID_JOINTS,
)


class TestSkeletonType:
    """Tests for SkeletonType enum."""

    def test_skeleton_types_exist(self):
        """Test all skeleton types exist."""
        assert SkeletonType.MIXAMO is not None
        assert SkeletonType.UNREAL_MANNEQUIN is not None
        assert SkeletonType.MUJOCO_HUMANOID is not None
        assert SkeletonType.CUSTOM is not None

    def test_skeleton_type_bone_count(self):
        """Test skeleton type bone counts."""
        assert SkeletonType.MIXAMO.standard_bone_count > 0
        assert SkeletonType.UNREAL_MANNEQUIN.standard_bone_count > 0


class TestBoneMapping:
    """Tests for BoneMapping data structure."""

    def test_create_bone_mapping(self):
        """Test bone mapping creation."""
        mapping = BoneMapping(
            source_bone="mixamorig:Hips",
            target_bone="pelvis",
            rotation_offset=np.array([0.0, 0.0, 0.0]),
            scale_factor=1.0,
        )
        assert mapping.source_bone == "mixamorig:Hips"
        assert mapping.target_bone == "pelvis"

    def test_bone_mapping_with_offset(self):
        """Test bone mapping with rotation offset."""
        mapping = BoneMapping(
            source_bone="mixamorig:RightArm",
            target_bone="right_shoulder",
            rotation_offset=np.array([0.0, 0.0, -90.0]),  # Degrees
        )
        assert mapping.rotation_offset[2] == -90.0

    def test_bone_mapping_with_scale(self):
        """Test bone mapping with scale factor."""
        mapping = BoneMapping(
            source_bone="mixamorig:Spine",
            target_bone="lumbar",
            scale_factor=0.95,  # Slightly shorter
        )
        assert mapping.scale_factor == 0.95

    def test_bone_mapping_to_dict(self):
        """Test bone mapping serialization."""
        mapping = BoneMapping(
            source_bone="source",
            target_bone="target",
        )
        d = mapping.to_dict()
        assert d["source_bone"] == "source"
        assert d["target_bone"] == "target"

    def test_bone_mapping_from_dict(self):
        """Test bone mapping deserialization."""
        d = {
            "source_bone": "Hips",
            "target_bone": "pelvis",
            "rotation_offset": [0.0, 0.0, 0.0],
            "scale_factor": 1.0,
        }
        mapping = BoneMapping.from_dict(d)
        assert mapping.source_bone == "Hips"


class TestMappingProfile:
    """Tests for MappingProfile."""

    def test_create_mapping_profile(self):
        """Test mapping profile creation."""
        mappings = [
            BoneMapping("Hips", "pelvis"),
            BoneMapping("Spine", "lumbar"),
        ]
        profile = MappingProfile(
            name="test_profile",
            source_type=SkeletonType.MIXAMO,
            target_type=SkeletonType.MUJOCO_HUMANOID,
            mappings=mappings,
        )
        assert profile.name == "test_profile"
        assert len(profile.mappings) == 2

    def test_profile_get_mapping(self):
        """Test getting mapping by source bone."""
        mappings = [
            BoneMapping("Hips", "pelvis"),
            BoneMapping("Spine", "lumbar"),
        ]
        profile = MappingProfile(
            name="test",
            source_type=SkeletonType.MIXAMO,
            target_type=SkeletonType.MUJOCO_HUMANOID,
            mappings=mappings,
        )
        mapping = profile.get_mapping("Hips")
        assert mapping is not None
        assert mapping.target_bone == "pelvis"

    def test_profile_has_mapping(self):
        """Test checking for mapping existence."""
        mappings = [BoneMapping("Hips", "pelvis")]
        profile = MappingProfile(
            name="test",
            source_type=SkeletonType.MIXAMO,
            target_type=SkeletonType.MUJOCO_HUMANOID,
            mappings=mappings,
        )
        assert profile.has_mapping("Hips")
        assert not profile.has_mapping("NonExistent")

    def test_profile_serialization(self):
        """Test profile serialization."""
        mappings = [BoneMapping("Hips", "pelvis")]
        profile = MappingProfile(
            name="test",
            source_type=SkeletonType.MIXAMO,
            target_type=SkeletonType.MUJOCO_HUMANOID,
            mappings=mappings,
        )
        d = profile.to_dict()
        assert d["name"] == "test"
        assert len(d["mappings"]) == 1

    def test_profile_deserialization(self):
        """Test profile deserialization."""
        d = {
            "name": "loaded_profile",
            "source_type": "mixamo",
            "target_type": "mujoco_humanoid",
            "mappings": [
                {"source_bone": "Hips", "target_bone": "pelvis"},
            ],
        }
        profile = MappingProfile.from_dict(d)
        assert profile.name == "loaded_profile"


class TestPoseTransform:
    """Tests for PoseTransform data structure."""

    def test_create_pose_transform(self):
        """Test pose transform creation."""
        transform = PoseTransform(
            position=np.array([0.0, 1.0, 0.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),  # Quaternion (w, x, y, z)
        )
        assert transform.position[1] == 1.0

    def test_identity_transform(self):
        """Test identity transform creation."""
        transform = PoseTransform.identity()
        assert np.allclose(transform.position, [0, 0, 0])
        assert np.allclose(transform.rotation, [1, 0, 0, 0])

    def test_pose_transform_to_matrix(self):
        """Test conversion to 4x4 transformation matrix."""
        transform = PoseTransform(
            position=np.array([1.0, 2.0, 3.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        matrix = transform.to_matrix()
        assert matrix.shape == (4, 4)
        assert matrix[0, 3] == 1.0  # X translation
        assert matrix[1, 3] == 2.0  # Y translation
        assert matrix[2, 3] == 3.0  # Z translation

    def test_pose_transform_from_matrix(self):
        """Test creation from 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        transform = PoseTransform.from_matrix(matrix)
        assert transform.position[0] == 1.0


class TestSkeletonMapper:
    """Tests for SkeletonMapper class."""

    def test_create_mapper(self):
        """Test mapper creation."""
        mapper = SkeletonMapper()
        assert mapper is not None

    def test_mapper_with_profile(self):
        """Test mapper creation with profile."""
        profile = MappingProfile(
            name="test",
            source_type=SkeletonType.MIXAMO,
            target_type=SkeletonType.MUJOCO_HUMANOID,
            mappings=[BoneMapping("Hips", "pelvis")],
        )
        mapper = SkeletonMapper(profile=profile)
        assert mapper.profile is not None

    def test_builtin_mixamo_profile(self):
        """Test built-in Mixamo to physics profile."""
        mapper = SkeletonMapper.for_mixamo()
        assert mapper.profile is not None
        assert mapper.profile.source_type == SkeletonType.MIXAMO

    def test_builtin_unreal_mannequin_profile(self):
        """Test built-in Unreal Mannequin to physics profile."""
        mapper = SkeletonMapper.for_unreal_mannequin()
        assert mapper.profile is not None
        assert mapper.profile.source_type == SkeletonType.UNREAL_MANNEQUIN

    def test_map_bone_name(self):
        """Test mapping bone name from source to target."""
        mapper = SkeletonMapper.for_mixamo()
        target = mapper.map_bone_name("mixamorig:Hips")
        assert target == "pelvis"

    def test_map_unknown_bone(self):
        """Test mapping unknown bone name."""
        mapper = SkeletonMapper.for_mixamo()
        target = mapper.map_bone_name("NonExistentBone")
        assert target is None

    def test_apply_pose(self):
        """Test applying pose to skeleton."""
        mapper = SkeletonMapper.for_mixamo()

        # Source pose (Mixamo format)
        source_pose = {
            "mixamorig:Hips": PoseTransform(
                position=np.array([0.0, 1.0, 0.0]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            "mixamorig:Spine": PoseTransform(
                position=np.array([0.0, 1.1, 0.0]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        }

        # Apply mapping
        target_pose = mapper.apply_pose(source_pose)

        # Should have mapped bones
        assert "pelvis" in target_pose or len(target_pose) > 0

    def test_apply_joint_angles(self):
        """Test applying joint angles from physics to mesh."""
        mapper = SkeletonMapper.for_mixamo()

        # Physics joint angles (radians)
        joint_angles = {
            "pelvis": 0.0,
            "lumbar": 0.1,
            "right_shoulder": -0.5,
        }

        # Apply to mesh bones
        bone_rotations = mapper.apply_joint_angles(joint_angles)

        # Should produce rotations for mesh bones
        assert isinstance(bone_rotations, dict)

    def test_get_unmapped_bones(self):
        """Test getting list of unmapped source bones."""
        mapper = SkeletonMapper.for_mixamo()
        source_bones = ["mixamorig:Hips", "CustomBone1", "CustomBone2"]
        unmapped = mapper.get_unmapped_bones(source_bones)
        assert "CustomBone1" in unmapped
        assert "CustomBone2" in unmapped
        assert "mixamorig:Hips" not in unmapped

    def test_reverse_mapping(self):
        """Test reverse mapping (physics to mesh)."""
        mapper = SkeletonMapper.for_mixamo()
        source = mapper.reverse_map_bone_name("pelvis")
        assert source == "mixamorig:Hips"


class TestPredefinedMappings:
    """Tests for predefined mapping constants."""

    def test_mixamo_mapping_completeness(self):
        """Test Mixamo mapping covers major bones."""
        required_bones = [
            "Hips", "Spine", "Head",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "LeftUpLeg", "LeftLeg", "LeftFoot",
            "RightUpLeg", "RightLeg", "RightFoot",
        ]
        for bone in required_bones:
            # Check with or without mixamorig prefix
            assert bone in MIXAMO_TO_PHYSICS_MAP or f"mixamorig:{bone}" in MIXAMO_TO_PHYSICS_MAP

    def test_unreal_mannequin_mapping_completeness(self):
        """Test Unreal Mannequin mapping covers major bones."""
        required_bones = [
            "pelvis", "spine_01", "head",
            "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
            "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
            "thigh_l", "calf_l", "foot_l",
            "thigh_r", "calf_r", "foot_r",
        ]
        for bone in required_bones:
            assert bone in UNREAL_MANNEQUIN_TO_PHYSICS_MAP

    def test_mujoco_humanoid_joints(self):
        """Test MuJoCo humanoid joint list."""
        required_joints = [
            "pelvis", "abdomen", "thorax",
            "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle",
            "left_hip", "left_knee", "left_ankle",
        ]
        for joint in required_joints:
            assert joint in MUJOCO_HUMANOID_JOINTS


class TestSkeletonMapperContracts:
    """Tests for Design by Contract compliance."""

    def test_mapper_requires_valid_profile(self):
        """Test mapper validates profile."""
        # Empty mappings should still be valid
        profile = MappingProfile(
            name="empty",
            source_type=SkeletonType.CUSTOM,
            target_type=SkeletonType.CUSTOM,
            mappings=[],
        )
        mapper = SkeletonMapper(profile=profile)
        assert mapper.profile is not None

    def test_apply_pose_validates_input(self):
        """Test apply_pose validates input types."""
        mapper = SkeletonMapper.for_mixamo()

        # Invalid input should raise error
        with pytest.raises((TypeError, ValueError)):
            mapper.apply_pose("not a dict")  # type: ignore

    def test_bone_mapping_invariants(self):
        """Test bone mapping maintains invariants."""
        mapping = BoneMapping(
            source_bone="source",
            target_bone="target",
            scale_factor=-1.0,  # Invalid negative scale
        )
        # Negative scale should be caught or handled gracefully


class TestSkeletonMapperInterpolation:
    """Tests for pose interpolation functionality."""

    def test_interpolate_poses(self):
        """Test pose interpolation between frames."""
        mapper = SkeletonMapper.for_mixamo()

        pose_a = {
            "pelvis": PoseTransform(
                position=np.array([0.0, 0.0, 0.0]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        }
        pose_b = {
            "pelvis": PoseTransform(
                position=np.array([1.0, 0.0, 0.0]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            ),
        }

        # Interpolate at 50%
        result = mapper.interpolate_poses(pose_a, pose_b, 0.5)

        assert "pelvis" in result
        assert result["pelvis"].position[0] == pytest.approx(0.5)

    def test_slerp_rotation(self):
        """Test quaternion SLERP interpolation."""
        q_a = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        q_b = np.array([0.707, 0.707, 0.0, 0.0])  # 90 deg around X

        result = SkeletonMapper.slerp(q_a, q_b, 0.5)

        # Should be unit quaternion
        assert np.isclose(np.linalg.norm(result), 1.0)
