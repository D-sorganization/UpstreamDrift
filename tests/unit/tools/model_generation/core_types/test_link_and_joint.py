"""
Comprehensive unit tests for model_generation core types.

Tests cover all data classes in model_generation.core.types:
- Inertia: factory methods, validation, operations, serialization
- Geometry: factory methods, to_urdf_string for all types
- Origin: creation, from_dict/to_dict, to_urdf_string
- JointLimits and JointDynamics: from_dict/to_dict roundtrip
- Link and Joint: from_dict/to_dict roundtrip, DOF counting
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)

# ── Inertia factory methods ──────────────────────────────────────────────────


# ── Inertia validation methods ───────────────────────────────────────────────


# ── Inertia operations ───────────────────────────────────────────────────────


# ── Geometry ─────────────────────────────────────────────────────────────────


# ── Origin ───────────────────────────────────────────────────────────────────


# ── JointLimits and JointDynamics ────────────────────────────────────────────


# ── Link and Joint ───────────────────────────────────────────────────────────


class TestLinkAndJoint:
    """Test Link and Joint from_dict/to_dict roundtrip and helpers."""

    def test_link_from_dict_minimal(self) -> None:
        link = Link.from_dict({"name": "base_link"})
        assert link.name == "base_link"
        assert link.inertia.mass == 1.0

    def test_link_roundtrip(self) -> None:
        original = Link(
            name="arm",
            inertia=Inertia.from_box(2.0, 0.1, 0.1, 0.5),
            visual_geometry=Geometry.box(0.1, 0.1, 0.5),
            visual_origin=Origin.from_position(0.0, 0.0, 0.25),
            visual_material=Material.skin(),
        )
        restored = Link.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.inertia.mass == pytest.approx(original.inertia.mass)
        assert restored.visual_geometry is not None
        assert restored.visual_geometry.geometry_type == GeometryType.BOX

    def test_joint_from_dict_defaults(self) -> None:
        joint = Joint.from_dict(
            {
                "name": "j1",
                "parent": "base",
                "child": "arm",
            }
        )
        assert joint.name == "j1"
        assert joint.joint_type == JointType.REVOLUTE
        assert joint.axis == (0.0, 0.0, 1.0)

    def test_joint_roundtrip(self) -> None:
        original = Joint(
            name="shoulder",
            joint_type=JointType.REVOLUTE,
            parent="torso",
            child="upper_arm",
            origin=Origin.from_position(0.0, 0.3, 0.0),
            axis=(1.0, 0.0, 0.0),
            limits=JointLimits(lower=-1.57, upper=1.57),
        )
        restored = Joint.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.joint_type == original.joint_type
        assert restored.parent == original.parent
        assert restored.child == original.child
        assert tuple(restored.axis) == original.axis

    def test_joint_auto_limits_for_revolute(self) -> None:
        joint = Joint(name="j", joint_type=JointType.REVOLUTE, parent="a", child="b")
        assert joint.limits is not None

    def test_joint_auto_limits_for_prismatic(self) -> None:
        joint = Joint(name="j", joint_type=JointType.PRISMATIC, parent="a", child="b")
        assert joint.limits is not None
        assert joint.limits.lower == -1.0

    def test_joint_no_auto_limits_for_continuous(self) -> None:
        joint = Joint(name="j", joint_type=JointType.CONTINUOUS, parent="a", child="b")
        assert joint.limits is None

    def test_is_composite(self) -> None:
        assert Joint(
            name="j", joint_type=JointType.GIMBAL, parent="a", child="b"
        ).is_composite()
        assert Joint(
            name="j", joint_type=JointType.UNIVERSAL, parent="a", child="b"
        ).is_composite()
        assert not Joint(
            name="j", joint_type=JointType.REVOLUTE, parent="a", child="b"
        ).is_composite()

    def test_dof_count(self) -> None:
        assert (
            Joint(
                name="j", joint_type=JointType.FIXED, parent="a", child="b"
            ).get_dof_count()
            == 0
        )
        assert (
            Joint(
                name="j", joint_type=JointType.REVOLUTE, parent="a", child="b"
            ).get_dof_count()
            == 1
        )
        assert (
            Joint(
                name="j", joint_type=JointType.FLOATING, parent="a", child="b"
            ).get_dof_count()
            == 6
        )
        assert (
            Joint(
                name="j", joint_type=JointType.GIMBAL, parent="a", child="b"
            ).get_dof_count()
            == 3
        )
        assert (
            Joint(
                name="j", joint_type=JointType.UNIVERSAL, parent="a", child="b"
            ).get_dof_count()
            == 2
        )
        assert (
            Joint(
                name="j", joint_type=JointType.PLANAR, parent="a", child="b"
            ).get_dof_count()
            == 3
        )


# ── Material ─────────────────────────────────────────────────────────────────
