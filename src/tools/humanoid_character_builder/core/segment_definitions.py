"""
Humanoid segment and joint definitions.

This module defines the hierarchical structure of humanoid body segments
and their connecting joints. These definitions are used to generate
the URDF kinematic tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JointType(Enum):
    """URDF joint types."""

    FIXED = "fixed"
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    FLOATING = "floating"
    PLANAR = "planar"

    # Composite joint types (expanded to multiple URDF joints)
    UNIVERSAL = "universal"  # 2 DOF
    GIMBAL = "gimbal"  # 3 DOF (ball joint approximation)
    SPHERICAL = "spherical"  # Alias for gimbal


class GeometryType(Enum):
    """Geometry types for visual and collision."""

    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    MESH = "mesh"


@dataclass
class JointLimits:
    """Joint limits specification."""

    lower: float = -3.14159  # radians
    upper: float = 3.14159
    effort: float = 100.0  # N*m
    velocity: float = 10.0  # rad/s

    def as_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "effort": self.effort,
            "velocity": self.velocity,
        }


@dataclass
class JointDefinition:
    """
    Definition of a joint connecting two segments.

    The joint connects the parent segment to the child segment.
    """

    name: str
    joint_type: JointType
    parent_segment: str
    child_segment: str

    # Joint axis (for revolute/prismatic)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Origin relative to parent frame
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Joint limits
    limits: JointLimits = field(default_factory=JointLimits)

    # Dynamics
    damping: float = 0.5
    friction: float = 0.0

    # For composite joints (universal, gimbal)
    secondary_axis: tuple[float, float, float] | None = None
    tertiary_axis: tuple[float, float, float] | None = None

    def is_composite(self) -> bool:
        """Check if this is a composite joint type."""
        return self.joint_type in (
            JointType.UNIVERSAL,
            JointType.GIMBAL,
            JointType.SPHERICAL,
        )

    def get_dof(self) -> int:
        """Get degrees of freedom for this joint."""
        dof_map = {
            JointType.FIXED: 0,
            JointType.REVOLUTE: 1,
            JointType.CONTINUOUS: 1,
            JointType.PRISMATIC: 1,
            JointType.UNIVERSAL: 2,
            JointType.GIMBAL: 3,
            JointType.SPHERICAL: 3,
            JointType.FLOATING: 6,
            JointType.PLANAR: 3,
        }
        return dof_map.get(self.joint_type, 1)


@dataclass
class GeometrySpec:
    """Geometry specification for visual or collision."""

    geometry_type: GeometryType

    # Dimensions (interpretation depends on type)
    # BOX: (x, y, z) size
    # CYLINDER: (radius, length)
    # SPHERE: (radius,)
    # CAPSULE: (radius, length)
    # MESH: not used (see mesh_path)
    dimensions: tuple[float, ...] = (0.1, 0.1, 0.1)

    # Mesh file path (for MESH type)
    mesh_path: str | None = None

    # Mesh scale
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Origin offset from link frame
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SegmentDefinition:
    """
    Definition of a body segment (URDF link).

    Contains all information needed to generate the URDF link element.
    """

    name: str

    # Parent segment (None for root)
    parent: str | None = None

    # Default mass (may be overridden by BodyParameters)
    default_mass_kg: float = 1.0

    # Mass ratio (fraction of total body mass, for scaling)
    mass_ratio: float = 0.01

    # Length ratio (fraction of total height)
    length_ratio: float = 0.1

    # Default geometry for visual
    visual_geometry: GeometrySpec = field(
        default_factory=lambda: GeometrySpec(GeometryType.BOX, (0.1, 0.1, 0.1))
    )

    # Default geometry for collision (can differ from visual)
    collision_geometry: GeometrySpec | None = None

    # Default color
    color_rgba: tuple[float, float, float, float] = (0.8, 0.7, 0.6, 1.0)

    # Center of mass offset from geometric center (as ratio of dimensions)
    com_offset_ratio: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Is this an end effector?
    is_end_effector: bool = False

    # Vertex group name for mesh segmentation (MakeHuman compatibility)
    vertex_group: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_collision_geometry(self) -> GeometrySpec:
        """Get collision geometry, defaulting to visual if not specified."""
        return self.collision_geometry or self.visual_geometry


# =============================================================================
# Standard Humanoid Segment Definitions
# =============================================================================

# Segment hierarchy (parent -> children mapping)
SEGMENT_HIERARCHY = {
    "pelvis": ["lumbar", "left_hip", "right_hip"],
    "lumbar": ["thorax"],
    "thorax": ["neck", "left_shoulder", "right_shoulder"],
    "neck": ["head"],
    "head": [],
    "left_shoulder": ["left_upper_arm"],
    "right_shoulder": ["right_upper_arm"],
    "left_upper_arm": ["left_forearm"],
    "right_upper_arm": ["right_forearm"],
    "left_forearm": ["left_hand"],
    "right_forearm": ["right_hand"],
    "left_hand": [],
    "right_hand": [],
    "left_hip": ["left_thigh"],
    "right_hip": ["right_thigh"],
    "left_thigh": ["left_shin"],
    "right_thigh": ["right_shin"],
    "left_shin": ["left_foot"],
    "right_shin": ["right_foot"],
    "left_foot": [],
    "right_foot": [],
}


def _create_segment_definitions() -> dict[str, SegmentDefinition]:
    """Create the standard humanoid segment definitions."""
    segments = {}

    # === Torso ===
    segments["pelvis"] = SegmentDefinition(
        name="pelvis",
        parent=None,
        mass_ratio=0.117,  # de Leva
        length_ratio=0.10,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.10, 0.15)),
        vertex_group="pelvis",
    )

    segments["lumbar"] = SegmentDefinition(
        name="lumbar",
        parent="pelvis",
        mass_ratio=0.139,
        length_ratio=0.10,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.09, 0.12)),
        vertex_group="spine-lower",
    )

    segments["thorax"] = SegmentDefinition(
        name="thorax",
        parent="lumbar",
        mass_ratio=0.179,
        length_ratio=0.12,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.12, 0.18)),
        vertex_group="spine-upper",
    )

    segments["neck"] = SegmentDefinition(
        name="neck",
        parent="thorax",
        mass_ratio=0.024,
        length_ratio=0.052,
        visual_geometry=GeometrySpec(GeometryType.CYLINDER, (0.04, 0.08)),
        vertex_group="neck",
    )

    segments["head"] = SegmentDefinition(
        name="head",
        parent="neck",
        mass_ratio=0.069,
        length_ratio=0.14,
        visual_geometry=GeometrySpec(GeometryType.SPHERE, (0.10,)),
        vertex_group="head",
        is_end_effector=True,
    )

    # === Shoulders ===
    segments["left_shoulder"] = SegmentDefinition(
        name="left_shoulder",
        parent="thorax",
        mass_ratio=0.015,
        length_ratio=0.06,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.04, 0.08)),
        vertex_group="shoulder.L",
    )

    segments["right_shoulder"] = SegmentDefinition(
        name="right_shoulder",
        parent="thorax",
        mass_ratio=0.015,
        length_ratio=0.06,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.04, 0.08)),
        vertex_group="shoulder.R",
    )

    # === Arms ===
    segments["left_upper_arm"] = SegmentDefinition(
        name="left_upper_arm",
        parent="left_shoulder",
        mass_ratio=0.027,
        length_ratio=0.186,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.035, 0.28)),
        vertex_group="upperarm.L",
    )

    segments["right_upper_arm"] = SegmentDefinition(
        name="right_upper_arm",
        parent="right_shoulder",
        mass_ratio=0.027,
        length_ratio=0.186,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.035, 0.28)),
        vertex_group="upperarm.R",
    )

    segments["left_forearm"] = SegmentDefinition(
        name="left_forearm",
        parent="left_upper_arm",
        mass_ratio=0.016,
        length_ratio=0.146,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.03, 0.22)),
        vertex_group="forearm.L",
    )

    segments["right_forearm"] = SegmentDefinition(
        name="right_forearm",
        parent="right_upper_arm",
        mass_ratio=0.016,
        length_ratio=0.146,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.03, 0.22)),
        vertex_group="forearm.R",
    )

    segments["left_hand"] = SegmentDefinition(
        name="left_hand",
        parent="left_forearm",
        mass_ratio=0.006,
        length_ratio=0.108,
        visual_geometry=GeometrySpec(GeometryType.BOX, (0.08, 0.04, 0.15)),
        vertex_group="hand.L",
        is_end_effector=True,
    )

    segments["right_hand"] = SegmentDefinition(
        name="right_hand",
        parent="right_forearm",
        mass_ratio=0.006,
        length_ratio=0.108,
        visual_geometry=GeometrySpec(GeometryType.BOX, (0.08, 0.04, 0.15)),
        vertex_group="hand.R",
        is_end_effector=True,
    )

    # === Hips (virtual segments for joint placement) ===
    segments["left_hip"] = SegmentDefinition(
        name="left_hip",
        parent="pelvis",
        mass_ratio=0.001,  # Minimal mass (virtual)
        length_ratio=0.0,
        visual_geometry=GeometrySpec(GeometryType.SPHERE, (0.02,)),
        vertex_group=None,
    )

    segments["right_hip"] = SegmentDefinition(
        name="right_hip",
        parent="pelvis",
        mass_ratio=0.001,
        length_ratio=0.0,
        visual_geometry=GeometrySpec(GeometryType.SPHERE, (0.02,)),
        vertex_group=None,
    )

    # === Legs ===
    segments["left_thigh"] = SegmentDefinition(
        name="left_thigh",
        parent="left_hip",
        mass_ratio=0.142,
        length_ratio=0.245,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.06, 0.40)),
        vertex_group="thigh.L",
    )

    segments["right_thigh"] = SegmentDefinition(
        name="right_thigh",
        parent="right_hip",
        mass_ratio=0.142,
        length_ratio=0.245,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.06, 0.40)),
        vertex_group="thigh.R",
    )

    segments["left_shin"] = SegmentDefinition(
        name="left_shin",
        parent="left_thigh",
        mass_ratio=0.043,
        length_ratio=0.246,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.04, 0.38)),
        vertex_group="shin.L",
    )

    segments["right_shin"] = SegmentDefinition(
        name="right_shin",
        parent="right_thigh",
        mass_ratio=0.043,
        length_ratio=0.246,
        visual_geometry=GeometrySpec(GeometryType.CAPSULE, (0.04, 0.38)),
        vertex_group="shin.R",
    )

    segments["left_foot"] = SegmentDefinition(
        name="left_foot",
        parent="left_shin",
        mass_ratio=0.014,
        length_ratio=0.152,
        visual_geometry=GeometrySpec(GeometryType.BOX, (0.08, 0.22, 0.05)),
        vertex_group="foot.L",
        is_end_effector=True,
    )

    segments["right_foot"] = SegmentDefinition(
        name="right_foot",
        parent="right_shin",
        mass_ratio=0.014,
        length_ratio=0.152,
        visual_geometry=GeometrySpec(GeometryType.BOX, (0.08, 0.22, 0.05)),
        vertex_group="foot.R",
        is_end_effector=True,
    )

    return segments


def _create_joint_definitions() -> dict[str, JointDefinition]:
    """Create the standard humanoid joint definitions."""
    joints = {}

    # === Spine Joints ===
    joints["pelvis_to_lumbar"] = JointDefinition(
        name="pelvis_to_lumbar",
        joint_type=JointType.UNIVERSAL,
        parent_segment="pelvis",
        child_segment="lumbar",
        axis=(1.0, 0.0, 0.0),
        secondary_axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, 0.08),
        limits=JointLimits(lower=-0.5, upper=0.5),
    )

    joints["lumbar_to_thorax"] = JointDefinition(
        name="lumbar_to_thorax",
        joint_type=JointType.UNIVERSAL,
        parent_segment="lumbar",
        child_segment="thorax",
        axis=(1.0, 0.0, 0.0),
        secondary_axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, 0.10),
        limits=JointLimits(lower=-0.4, upper=0.4),
    )

    joints["thorax_to_neck"] = JointDefinition(
        name="thorax_to_neck",
        joint_type=JointType.UNIVERSAL,
        parent_segment="thorax",
        child_segment="neck",
        axis=(1.0, 0.0, 0.0),
        secondary_axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, 0.15),
        limits=JointLimits(lower=-0.5, upper=0.5),
    )

    joints["neck_to_head"] = JointDefinition(
        name="neck_to_head",
        joint_type=JointType.GIMBAL,
        parent_segment="neck",
        child_segment="head",
        axis=(0.0, 0.0, 1.0),
        secondary_axis=(0.0, 1.0, 0.0),
        tertiary_axis=(1.0, 0.0, 0.0),
        origin_xyz=(0.0, 0.0, 0.06),
        limits=JointLimits(lower=-1.0, upper=1.0),
    )

    # === Shoulder Joints ===
    joints["thorax_to_left_shoulder"] = JointDefinition(
        name="thorax_to_left_shoulder",
        joint_type=JointType.FIXED,
        parent_segment="thorax",
        child_segment="left_shoulder",
        origin_xyz=(0.15, 0.0, 0.12),
    )

    joints["thorax_to_right_shoulder"] = JointDefinition(
        name="thorax_to_right_shoulder",
        joint_type=JointType.FIXED,
        parent_segment="thorax",
        child_segment="right_shoulder",
        origin_xyz=(-0.15, 0.0, 0.12),
    )

    # === Arm Joints ===
    joints["left_shoulder_to_upper_arm"] = JointDefinition(
        name="left_shoulder_to_upper_arm",
        joint_type=JointType.GIMBAL,
        parent_segment="left_shoulder",
        child_segment="left_upper_arm",
        axis=(0.0, 0.0, 1.0),
        secondary_axis=(0.0, 1.0, 0.0),
        tertiary_axis=(1.0, 0.0, 0.0),
        origin_xyz=(0.06, 0.0, 0.0),
        limits=JointLimits(lower=-2.5, upper=2.5),
    )

    joints["right_shoulder_to_upper_arm"] = JointDefinition(
        name="right_shoulder_to_upper_arm",
        joint_type=JointType.GIMBAL,
        parent_segment="right_shoulder",
        child_segment="right_upper_arm",
        axis=(0.0, 0.0, 1.0),
        secondary_axis=(0.0, 1.0, 0.0),
        tertiary_axis=(1.0, 0.0, 0.0),
        origin_xyz=(-0.06, 0.0, 0.0),
        limits=JointLimits(lower=-2.5, upper=2.5),
    )

    joints["left_elbow"] = JointDefinition(
        name="left_elbow",
        joint_type=JointType.REVOLUTE,
        parent_segment="left_upper_arm",
        child_segment="left_forearm",
        axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, -0.28),
        limits=JointLimits(lower=0.0, upper=2.5),
    )

    joints["right_elbow"] = JointDefinition(
        name="right_elbow",
        joint_type=JointType.REVOLUTE,
        parent_segment="right_upper_arm",
        child_segment="right_forearm",
        axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, -0.28),
        limits=JointLimits(lower=0.0, upper=2.5),
    )

    joints["left_wrist"] = JointDefinition(
        name="left_wrist",
        joint_type=JointType.UNIVERSAL,
        parent_segment="left_forearm",
        child_segment="left_hand",
        axis=(1.0, 0.0, 0.0),
        secondary_axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, -0.22),
        limits=JointLimits(lower=-1.2, upper=1.2),
    )

    joints["right_wrist"] = JointDefinition(
        name="right_wrist",
        joint_type=JointType.UNIVERSAL,
        parent_segment="right_forearm",
        child_segment="right_hand",
        axis=(1.0, 0.0, 0.0),
        secondary_axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, -0.22),
        limits=JointLimits(lower=-1.2, upper=1.2),
    )

    # === Hip Joints ===
    joints["pelvis_to_left_hip"] = JointDefinition(
        name="pelvis_to_left_hip",
        joint_type=JointType.FIXED,
        parent_segment="pelvis",
        child_segment="left_hip",
        origin_xyz=(0.09, 0.0, -0.05),
    )

    joints["pelvis_to_right_hip"] = JointDefinition(
        name="pelvis_to_right_hip",
        joint_type=JointType.FIXED,
        parent_segment="pelvis",
        child_segment="right_hip",
        origin_xyz=(-0.09, 0.0, -0.05),
    )

    # === Leg Joints ===
    joints["left_hip_joint"] = JointDefinition(
        name="left_hip_joint",
        joint_type=JointType.GIMBAL,
        parent_segment="left_hip",
        child_segment="left_thigh",
        axis=(0.0, 0.0, 1.0),
        secondary_axis=(1.0, 0.0, 0.0),
        tertiary_axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, 0.0),
        limits=JointLimits(lower=-2.0, upper=2.0),
    )

    joints["right_hip_joint"] = JointDefinition(
        name="right_hip_joint",
        joint_type=JointType.GIMBAL,
        parent_segment="right_hip",
        child_segment="right_thigh",
        axis=(0.0, 0.0, 1.0),
        secondary_axis=(1.0, 0.0, 0.0),
        tertiary_axis=(0.0, 1.0, 0.0),
        origin_xyz=(0.0, 0.0, 0.0),
        limits=JointLimits(lower=-2.0, upper=2.0),
    )

    joints["left_knee"] = JointDefinition(
        name="left_knee",
        joint_type=JointType.REVOLUTE,
        parent_segment="left_thigh",
        child_segment="left_shin",
        axis=(1.0, 0.0, 0.0),
        origin_xyz=(0.0, 0.0, -0.40),
        limits=JointLimits(lower=-2.5, upper=0.0),
    )

    joints["right_knee"] = JointDefinition(
        name="right_knee",
        joint_type=JointType.REVOLUTE,
        parent_segment="right_thigh",
        child_segment="right_shin",
        axis=(1.0, 0.0, 0.0),
        origin_xyz=(0.0, 0.0, -0.40),
        limits=JointLimits(lower=-2.5, upper=0.0),
    )

    joints["left_ankle"] = JointDefinition(
        name="left_ankle",
        joint_type=JointType.UNIVERSAL,
        parent_segment="left_shin",
        child_segment="left_foot",
        axis=(1.0, 0.0, 0.0),
        secondary_axis=(0.0, 0.0, 1.0),
        origin_xyz=(0.0, 0.0, -0.38),
        limits=JointLimits(lower=-0.8, upper=0.8),
    )

    joints["right_ankle"] = JointDefinition(
        name="right_ankle",
        joint_type=JointType.UNIVERSAL,
        parent_segment="right_shin",
        child_segment="right_foot",
        axis=(1.0, 0.0, 0.0),
        secondary_axis=(0.0, 0.0, 1.0),
        origin_xyz=(0.0, 0.0, -0.38),
        limits=JointLimits(lower=-0.8, upper=0.8),
    )

    return joints


# Module-level constants
HUMANOID_SEGMENTS: dict[str, SegmentDefinition] = _create_segment_definitions()
HUMANOID_JOINTS: dict[str, JointDefinition] = _create_joint_definitions()


def get_segment(name: str) -> SegmentDefinition | None:
    """Get segment definition by name."""
    return HUMANOID_SEGMENTS.get(name)


def get_joint(name: str) -> JointDefinition | None:
    """Get joint definition by name."""
    return HUMANOID_JOINTS.get(name)


def get_all_segment_names() -> list[str]:
    """Get list of all segment names."""
    return list(HUMANOID_SEGMENTS.keys())


def get_children(segment_name: str) -> list[str]:
    """Get child segment names for a given segment."""
    return SEGMENT_HIERARCHY.get(segment_name, [])


def get_segment_chain(segment_name: str) -> list[str]:
    """Get chain from root to specified segment."""
    chain = []
    current = segment_name
    while current is not None:
        chain.insert(0, current)
        segment = HUMANOID_SEGMENTS.get(current)
        current = segment.parent if segment else None
    return chain
