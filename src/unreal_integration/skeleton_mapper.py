"""Skeleton mapping system for gaming skeletons to physics models.

This module provides mapping between gaming industry skeleton formats
(Mixamo, Unreal Mannequin) and physics simulation joint hierarchies
(MuJoCo, Drake, Pinocchio).

Design by Contract:
    - Mappings maintain bone hierarchy consistency
    - Transformations preserve pose integrity
    - Unknown bones are handled gracefully

Supported Skeleton Types:
    - Mixamo (Adobe)
    - Unreal Engine Mannequin
    - MuJoCo Humanoid
    - Custom user-defined

Usage:
    from src.unreal_integration.skeleton_mapper import SkeletonMapper

    # Create mapper for Mixamo to physics
    mapper = SkeletonMapper.for_mixamo()

    # Map a pose
    physics_pose = mapper.apply_pose(mixamo_pose)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Predefined Mappings
# ============================================================================

# Mixamo skeleton bone names to physics joint names
MIXAMO_TO_PHYSICS_MAP: dict[str, str] = {
    # Root
    "mixamorig:Hips": "pelvis",
    "Hips": "pelvis",
    # Spine
    "mixamorig:Spine": "lumbar",
    "mixamorig:Spine1": "thorax_lower",
    "mixamorig:Spine2": "thorax_upper",
    "Spine": "lumbar",
    "Spine1": "thorax_lower",
    "Spine2": "thorax_upper",
    # Head
    "mixamorig:Neck": "neck",
    "mixamorig:Head": "head",
    "Neck": "neck",
    "Head": "head",
    # Left Arm
    "mixamorig:LeftShoulder": "left_clavicle",
    "mixamorig:LeftArm": "left_shoulder",
    "mixamorig:LeftForeArm": "left_elbow",
    "mixamorig:LeftHand": "left_wrist",
    "LeftShoulder": "left_clavicle",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    # Right Arm
    "mixamorig:RightShoulder": "right_clavicle",
    "mixamorig:RightArm": "right_shoulder",
    "mixamorig:RightForeArm": "right_elbow",
    "mixamorig:RightHand": "right_wrist",
    "RightShoulder": "right_clavicle",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
    # Left Leg
    "mixamorig:LeftUpLeg": "left_hip",
    "mixamorig:LeftLeg": "left_knee",
    "mixamorig:LeftFoot": "left_ankle",
    "mixamorig:LeftToeBase": "left_toe",
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToeBase": "left_toe",
    # Right Leg
    "mixamorig:RightUpLeg": "right_hip",
    "mixamorig:RightLeg": "right_knee",
    "mixamorig:RightFoot": "right_ankle",
    "mixamorig:RightToeBase": "right_toe",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToeBase": "right_toe",
    # Fingers (simplified)
    "mixamorig:LeftHandIndex1": "left_index",
    "mixamorig:RightHandIndex1": "right_index",
}

# Unreal Engine Mannequin bone names to physics joint names
UNREAL_MANNEQUIN_TO_PHYSICS_MAP: dict[str, str] = {
    # Root
    "pelvis": "pelvis",
    "root": "pelvis",
    # Spine
    "spine_01": "lumbar",
    "spine_02": "thorax_lower",
    "spine_03": "thorax_upper",
    # Head
    "neck_01": "neck",
    "head": "head",
    # Left Arm
    "clavicle_l": "left_clavicle",
    "upperarm_l": "left_shoulder",
    "lowerarm_l": "left_elbow",
    "hand_l": "left_wrist",
    # Right Arm
    "clavicle_r": "right_clavicle",
    "upperarm_r": "right_shoulder",
    "lowerarm_r": "right_elbow",
    "hand_r": "right_wrist",
    # Left Leg
    "thigh_l": "left_hip",
    "calf_l": "left_knee",
    "foot_l": "left_ankle",
    "ball_l": "left_toe",
    # Right Leg
    "thigh_r": "right_hip",
    "calf_r": "right_knee",
    "foot_r": "right_ankle",
    "ball_r": "right_toe",
}

# MuJoCo humanoid joint names
MUJOCO_HUMANOID_JOINTS: list[str] = [
    "pelvis",
    "abdomen",
    "thorax",
    "head",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
]


class SkeletonType(Enum):
    """Supported skeleton types."""

    MIXAMO = auto()
    UNREAL_MANNEQUIN = auto()
    MUJOCO_HUMANOID = auto()
    PINOCCHIO = auto()
    OPENSIM = auto()
    CUSTOM = auto()

    @property
    def standard_bone_count(self) -> int:
        """Get standard bone count for skeleton type."""
        counts = {
            SkeletonType.MIXAMO: 65,
            SkeletonType.UNREAL_MANNEQUIN: 68,
            SkeletonType.MUJOCO_HUMANOID: 16,
            SkeletonType.PINOCCHIO: 24,
            SkeletonType.OPENSIM: 28,
            SkeletonType.CUSTOM: 0,
        }
        return counts.get(self, 0)


@dataclass
class BoneMapping:
    """Mapping between source and target skeleton bones.

    Attributes:
        source_bone: Name of bone in source skeleton.
        target_bone: Name of bone/joint in target skeleton.
        rotation_offset: Rotation offset in degrees (x, y, z).
        scale_factor: Scale factor for bone length.
        position_offset: Position offset (optional).
    """

    source_bone: str
    target_bone: str
    rotation_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale_factor: float = 1.0
    position_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        """Ensure arrays are numpy."""
        if not isinstance(self.rotation_offset, np.ndarray):
            self.rotation_offset = np.array(self.rotation_offset)
        if not isinstance(self.position_offset, np.ndarray):
            self.position_offset = np.array(self.position_offset)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "source_bone": self.source_bone,
            "target_bone": self.target_bone,
            "rotation_offset": self.rotation_offset.tolist(),
            "scale_factor": self.scale_factor,
            "position_offset": self.position_offset.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BoneMapping:
        """Create from dictionary.

        Args:
            d: Dictionary representation.

        Returns:
            New BoneMapping instance.
        """
        return cls(
            source_bone=d["source_bone"],
            target_bone=d["target_bone"],
            rotation_offset=np.array(d.get("rotation_offset", [0, 0, 0])),
            scale_factor=d.get("scale_factor", 1.0),
            position_offset=np.array(d.get("position_offset", [0, 0, 0])),
        )


@dataclass
class MappingProfile:
    """Complete skeleton mapping profile.

    Attributes:
        name: Profile name.
        source_type: Source skeleton type.
        target_type: Target skeleton type.
        mappings: List of bone mappings.
        description: Optional description.
    """

    name: str
    source_type: SkeletonType
    target_type: SkeletonType
    mappings: list[BoneMapping]
    description: str = ""

    def __post_init__(self) -> None:
        """Build lookup table."""
        self._source_to_target: dict[str, BoneMapping] = {
            m.source_bone: m for m in self.mappings
        }
        # Build reverse mapping, preferring longer/qualified source names
        self._target_to_source: dict[str, BoneMapping] = {}
        for m in self.mappings:
            existing = self._target_to_source.get(m.target_bone)
            if existing is None or len(m.source_bone) > len(existing.source_bone):
                self._target_to_source[m.target_bone] = m

    def get_mapping(self, source_bone: str) -> BoneMapping | None:
        """Get mapping for source bone.

        Args:
            source_bone: Source bone name.

        Returns:
            BoneMapping if found, None otherwise.
        """
        return self._source_to_target.get(source_bone)

    def get_reverse_mapping(self, target_bone: str) -> BoneMapping | None:
        """Get mapping for target bone (reverse lookup).

        Args:
            target_bone: Target bone name.

        Returns:
            BoneMapping if found, None otherwise.
        """
        return self._target_to_source.get(target_bone)

    def has_mapping(self, source_bone: str) -> bool:
        """Check if source bone has mapping.

        Args:
            source_bone: Source bone name.

        Returns:
            True if mapping exists.
        """
        return source_bone in self._source_to_target

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "source_type": self.source_type.name.lower(),
            "target_type": self.target_type.name.lower(),
            "mappings": [m.to_dict() for m in self.mappings],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MappingProfile:
        """Create from dictionary.

        Args:
            d: Dictionary representation.

        Returns:
            New MappingProfile instance.
        """
        source_type = SkeletonType[d["source_type"].upper()]
        target_type = SkeletonType[d["target_type"].upper()]
        mappings = [BoneMapping.from_dict(m) for m in d["mappings"]]

        return cls(
            name=d["name"],
            source_type=source_type,
            target_type=target_type,
            mappings=mappings,
            description=d.get("description", ""),
        )


@dataclass
class PoseTransform:
    """Pose transformation for a single bone/joint.

    Attributes:
        position: 3D position.
        rotation: Quaternion rotation (w, x, y, z).
        scale: Optional scale (default 1.0).
    """

    position: np.ndarray
    rotation: np.ndarray
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Ensure arrays are numpy."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        if not isinstance(self.rotation, np.ndarray):
            self.rotation = np.array(self.rotation)

    @classmethod
    def identity(cls) -> PoseTransform:
        """Create identity transform.

        Returns:
            Identity PoseTransform.
        """
        return cls(
            position=np.zeros(3),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        )

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix.

        Returns:
            4x4 transformation matrix.
        """
        # Build rotation matrix from quaternion
        w, x, y, z = self.rotation

        # Rotation matrix from quaternion
        rot = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

        # Build 4x4 matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rot * self.scale
        matrix[:3, 3] = self.position

        return matrix

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> PoseTransform:
        """Create from 4x4 transformation matrix.

        Args:
            matrix: 4x4 transformation matrix.

        Returns:
            New PoseTransform instance.
        """
        position = matrix[:3, 3].copy()

        # Extract rotation (assume no skew)
        rot = matrix[:3, :3].copy()
        scale = float(np.linalg.norm(rot[:, 0]))
        rot = rot / scale

        # Convert rotation matrix to quaternion
        trace = np.trace(rot)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot[2, 1] - rot[1, 2]) * s
            y = (rot[0, 2] - rot[2, 0]) * s
            z = (rot[1, 0] - rot[0, 1]) * s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s

        rotation = np.array([w, x, y, z])

        return cls(position=position, rotation=rotation, scale=scale)


class SkeletonMapper:
    """Maps poses between different skeleton formats.

    Design by Contract:
        Preconditions:
            - apply_pose requires valid pose dictionary
            - Bone names must match profile mappings

        Postconditions:
            - Output pose has same number of mapped bones
            - Unmapped bones are excluded from output

    Example:
        >>> mapper = SkeletonMapper.for_mixamo()
        >>> physics_pose = mapper.apply_pose(mixamo_pose)
    """

    def __init__(self, profile: MappingProfile | None = None) -> None:
        """Initialize skeleton mapper.

        Args:
            profile: Mapping profile (optional).
        """
        self.profile = profile

    @classmethod
    def for_mixamo(cls) -> SkeletonMapper:
        """Create mapper for Mixamo to physics.

        Returns:
            SkeletonMapper configured for Mixamo.
        """
        mappings = [
            BoneMapping(source, target)
            for source, target in MIXAMO_TO_PHYSICS_MAP.items()
        ]
        profile = MappingProfile(
            name="mixamo_to_physics",
            source_type=SkeletonType.MIXAMO,
            target_type=SkeletonType.MUJOCO_HUMANOID,
            mappings=mappings,
            description="Maps Mixamo skeleton to physics simulation joints",
        )
        return cls(profile=profile)

    @classmethod
    def for_unreal_mannequin(cls) -> SkeletonMapper:
        """Create mapper for Unreal Mannequin to physics.

        Returns:
            SkeletonMapper configured for Unreal Mannequin.
        """
        mappings = [
            BoneMapping(source, target)
            for source, target in UNREAL_MANNEQUIN_TO_PHYSICS_MAP.items()
        ]
        profile = MappingProfile(
            name="unreal_mannequin_to_physics",
            source_type=SkeletonType.UNREAL_MANNEQUIN,
            target_type=SkeletonType.MUJOCO_HUMANOID,
            mappings=mappings,
            description="Maps Unreal Engine Mannequin to physics simulation joints",
        )
        return cls(profile=profile)

    def map_bone_name(self, source_bone: str) -> str | None:
        """Map source bone name to target.

        Args:
            source_bone: Source skeleton bone name.

        Returns:
            Target skeleton bone name, or None if not mapped.
        """
        if self.profile is None:
            return None
        mapping = self.profile.get_mapping(source_bone)
        return mapping.target_bone if mapping else None

    def reverse_map_bone_name(self, target_bone: str) -> str | None:
        """Map target bone name to source (reverse).

        Args:
            target_bone: Target skeleton bone name.

        Returns:
            Source skeleton bone name, or None if not mapped.
        """
        if self.profile is None:
            return None
        mapping = self.profile.get_reverse_mapping(target_bone)
        return mapping.source_bone if mapping else None

    def apply_pose(
        self, source_pose: dict[str, PoseTransform]
    ) -> dict[str, PoseTransform]:
        """Apply pose mapping from source to target skeleton.

        Args:
            source_pose: Dictionary of bone name to PoseTransform.

        Returns:
            Dictionary of target bone name to transformed PoseTransform.

        Raises:
            TypeError: If source_pose is not a dictionary.
        """
        if not isinstance(source_pose, dict):
            raise TypeError("source_pose must be a dictionary")

        if self.profile is None:
            return source_pose

        target_pose: dict[str, PoseTransform] = {}

        for source_bone, transform in source_pose.items():
            mapping = self.profile.get_mapping(source_bone)
            if mapping is None:
                continue

            # Apply rotation offset
            if np.any(mapping.rotation_offset != 0):
                # Convert offset from degrees to radians
                offset_rad = np.radians(mapping.rotation_offset)
                offset_quat = self._euler_to_quaternion(*offset_rad)
                new_rotation = self._quaternion_multiply(
                    transform.rotation, offset_quat
                )
            else:
                new_rotation = transform.rotation

            # Apply position offset and scale
            new_position = (
                transform.position * mapping.scale_factor + mapping.position_offset
            )

            target_pose[mapping.target_bone] = PoseTransform(
                position=new_position,
                rotation=new_rotation,
                scale=transform.scale * mapping.scale_factor,
            )

        return target_pose

    def apply_joint_angles(
        self,
        joint_angles: dict[str, float],
        axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> dict[str, np.ndarray]:
        """Apply joint angles from physics to mesh bone rotations.

        Args:
            joint_angles: Dictionary of joint name to angle (radians).
            axis: Rotation axis (default Z-axis).

        Returns:
            Dictionary of mesh bone name to rotation quaternion.
        """
        if self.profile is None:
            return {}

        bone_rotations: dict[str, np.ndarray] = {}

        for target_bone, angle in joint_angles.items():
            mapping = self.profile.get_reverse_mapping(target_bone)
            if mapping is None:
                continue

            # Create rotation quaternion from axis-angle
            half_angle = angle / 2.0
            quat = np.array(
                [
                    np.cos(half_angle),
                    axis[0] * np.sin(half_angle),
                    axis[1] * np.sin(half_angle),
                    axis[2] * np.sin(half_angle),
                ]
            )

            bone_rotations[mapping.source_bone] = quat

        return bone_rotations

    def get_unmapped_bones(self, source_bones: list[str]) -> list[str]:
        """Get list of source bones without mappings.

        Args:
            source_bones: List of source bone names.

        Returns:
            List of unmapped bone names.
        """
        if self.profile is None:
            return source_bones

        return [bone for bone in source_bones if not self.profile.has_mapping(bone)]

    def interpolate_poses(
        self,
        pose_a: dict[str, PoseTransform],
        pose_b: dict[str, PoseTransform],
        t: float,
    ) -> dict[str, PoseTransform]:
        """Interpolate between two poses.

        Args:
            pose_a: Starting pose.
            pose_b: Ending pose.
            t: Interpolation factor (0-1).

        Returns:
            Interpolated pose.
        """
        result: dict[str, PoseTransform] = {}

        # Get all bone names
        bones = set(pose_a.keys()) | set(pose_b.keys())

        for bone in bones:
            if bone in pose_a and bone in pose_b:
                # Interpolate position
                pos = pose_a[bone].position * (1 - t) + pose_b[bone].position * t

                # SLERP rotation
                rot = self.slerp(pose_a[bone].rotation, pose_b[bone].rotation, t)

                # Interpolate scale
                scale = pose_a[bone].scale * (1 - t) + pose_b[bone].scale * t

                result[bone] = PoseTransform(position=pos, rotation=rot, scale=scale)
            elif bone in pose_a:
                result[bone] = pose_a[bone]
            else:
                result[bone] = pose_b[bone]

        return result

    @staticmethod
    def slerp(q_a: np.ndarray, q_b: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions.

        Args:
            q_a: Starting quaternion.
            q_b: Ending quaternion.
            t: Interpolation factor (0-1).

        Returns:
            Interpolated quaternion.
        """
        # Normalize inputs
        q_a = q_a / np.linalg.norm(q_a)
        q_b = q_b / np.linalg.norm(q_b)

        # Compute dot product
        dot = np.dot(q_a, q_b)

        # If negative dot, negate one quaternion to take shorter path
        if dot < 0:
            q_b = -q_b
            dot = -dot

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q_a + t * (q_b - q_a)
            return result / np.linalg.norm(result)

        # SLERP formula
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return s0 * q_a + s1 * q_b

    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to quaternion.

        Args:
            roll: Rotation around X axis (radians).
            pitch: Rotation around Y axis (radians).
            yaw: Rotation around Z axis (radians).

        Returns:
            Quaternion as (w, x, y, z).
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        return np.array(
            [
                cr * cp * cy + sr * sp * sy,  # w
                sr * cp * cy - cr * sp * sy,  # x
                cr * sp * cy + sr * cp * sy,  # y
                cr * cp * sy - sr * sp * cy,  # z
            ]
        )

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions.

        Args:
            q1: First quaternion (w, x, y, z).
            q2: Second quaternion (w, x, y, z).

        Returns:
            Product quaternion.
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
