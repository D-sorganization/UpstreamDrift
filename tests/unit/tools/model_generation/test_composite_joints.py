"""Tests for composite joint expansion utilities.

These tests verify the shared functions ``expand_gimbal_joint`` and
``expand_universal_joint`` that decompose multi-DOF joints into
URDF-compatible single-axis revolute chains.
"""

from __future__ import annotations

import math

import pytest
from model_generation.core.composite_joints import (
    expand_gimbal_joint,
    expand_universal_joint,
)
from model_generation.core.constants import INTERMEDIATE_LINK_MASS
from model_generation.core.types import (
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Origin,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gimbal_joint() -> Joint:
    """A basic gimbal joint for testing."""
    return Joint(
        name="shoulder",
        joint_type=JointType.GIMBAL,
        parent="torso",
        child="upper_arm",
        origin=Origin(xyz=(0.0, 0.2, 0.0)),
        dynamics=JointDynamics(damping=1.0, friction=0.1),
    )


@pytest.fixture
def universal_joint() -> Joint:
    """A basic universal joint for testing."""
    return Joint(
        name="wrist",
        joint_type=JointType.UNIVERSAL,
        parent="forearm",
        child="hand",
        origin=Origin(xyz=(0.0, 0.0, -0.25)),
        dynamics=JointDynamics(damping=0.5, friction=0.05),
    )


@pytest.fixture
def gimbal_joint_custom_axes() -> Joint:
    """A gimbal joint with custom axes and per-DOF limits."""
    return Joint(
        name="hip",
        joint_type=JointType.GIMBAL,
        parent="pelvis",
        child="thigh",
        origin=Origin(xyz=(0.1, 0.0, -0.1)),
        composite_axes=[(1, 0, 0), (0, 0, 1), (0, 1, 0)],
        composite_limits=[
            JointLimits(lower=-1.5, upper=1.5, effort=500.0, velocity=5.0),
            JointLimits(lower=-0.5, upper=0.5, effort=300.0, velocity=3.0),
            JointLimits(lower=-2.0, upper=0.5, effort=400.0, velocity=4.0),
        ],
        dynamics=JointDynamics(damping=2.0, friction=0.2),
    )


@pytest.fixture
def universal_joint_custom_axes() -> Joint:
    """A universal joint with custom axes and per-DOF limits."""
    return Joint(
        name="ankle",
        joint_type=JointType.UNIVERSAL,
        parent="shin",
        child="foot",
        origin=Origin(xyz=(0.0, 0.0, -0.4)),
        composite_axes=[(0, 1, 0), (1, 0, 0)],
        composite_limits=[
            JointLimits(lower=-0.8, upper=0.5, effort=200.0, velocity=6.0),
            JointLimits(lower=-0.3, upper=0.3, effort=150.0, velocity=4.0),
        ],
        dynamics=JointDynamics(damping=0.3, friction=0.0),
    )


# ---------------------------------------------------------------------------
# Gimbal joint expansion tests
# ---------------------------------------------------------------------------


class TestExpandGimbalJoint:
    """Tests for expand_gimbal_joint()."""

    def test_produces_two_intermediate_links(self, gimbal_joint: Joint) -> None:
        """Gimbal expansion should produce exactly 2 intermediate links."""
        links, _ = expand_gimbal_joint(gimbal_joint)
        assert len(links) == 2

    def test_produces_three_revolute_joints(self, gimbal_joint: Joint) -> None:
        """Gimbal expansion should produce exactly 3 revolute joints."""
        _, joints = expand_gimbal_joint(gimbal_joint)
        assert len(joints) == 3

    def test_all_joints_are_revolute(self, gimbal_joint: Joint) -> None:
        """All expanded joints should be of type REVOLUTE."""
        _, joints = expand_gimbal_joint(gimbal_joint)
        for j in joints:
            assert j.joint_type == JointType.REVOLUTE

    def test_intermediate_link_names(self, gimbal_joint: Joint) -> None:
        """Intermediate links should be named consistently."""
        links, _ = expand_gimbal_joint(gimbal_joint)
        assert links[0].name == "shoulder_intermediate_1"
        assert links[1].name == "shoulder_intermediate_2"

    def test_intermediate_link_mass(self, gimbal_joint: Joint) -> None:
        """Intermediate links should have the standard intermediate mass."""
        links, _ = expand_gimbal_joint(gimbal_joint)
        for link in links:
            assert link.inertia.mass == INTERMEDIATE_LINK_MASS

    def test_intermediate_link_inertia_small(self, gimbal_joint: Joint) -> None:
        """Intermediate links should have negligible inertia values."""
        links, _ = expand_gimbal_joint(gimbal_joint)
        for link in links:
            assert link.inertia.ixx == pytest.approx(1e-6)
            assert link.inertia.iyy == pytest.approx(1e-6)
            assert link.inertia.izz == pytest.approx(1e-6)

    def test_parent_child_connectivity(self, gimbal_joint: Joint) -> None:
        """The chain should be: parent -> int1 -> int2 -> child."""
        links, joints = expand_gimbal_joint(gimbal_joint)
        # Joint 1: parent -> intermediate_1
        assert joints[0].parent == "torso"
        assert joints[0].child == "shoulder_intermediate_1"
        # Joint 2: intermediate_1 -> intermediate_2
        assert joints[1].parent == "shoulder_intermediate_1"
        assert joints[1].child == "shoulder_intermediate_2"
        # Joint 3: intermediate_2 -> child
        assert joints[2].parent == "shoulder_intermediate_2"
        assert joints[2].child == "upper_arm"

    def test_default_axes_zyx(self, gimbal_joint: Joint) -> None:
        """Default axes should be Z-Y-X Euler sequence."""
        _, joints = expand_gimbal_joint(gimbal_joint)
        assert joints[0].axis == (0, 0, 1)  # Z
        assert joints[1].axis == (0, 1, 0)  # Y
        assert joints[2].axis == (1, 0, 0)  # X

    def test_custom_axes(self, gimbal_joint_custom_axes: Joint) -> None:
        """Custom composite_axes should be respected."""
        _, joints = expand_gimbal_joint(gimbal_joint_custom_axes)
        assert joints[0].axis == (1, 0, 0)
        assert joints[1].axis == (0, 0, 1)
        assert joints[2].axis == (0, 1, 0)

    def test_origin_only_on_first_joint(self, gimbal_joint: Joint) -> None:
        """Only the first joint should carry the original origin."""
        _, joints = expand_gimbal_joint(gimbal_joint)
        assert joints[0].origin.xyz == (0.0, 0.2, 0.0)
        assert joints[1].origin.xyz == (0.0, 0.0, 0.0)
        assert joints[2].origin.xyz == (0.0, 0.0, 0.0)

    def test_dynamics_propagated(self, gimbal_joint: Joint) -> None:
        """Dynamics should propagate to all expanded joints."""
        _, joints = expand_gimbal_joint(gimbal_joint)
        for j in joints:
            assert j.dynamics.damping == 1.0
            assert j.dynamics.friction == 0.1

    def test_limits_propagation_default(self, gimbal_joint: Joint) -> None:
        """When no composite_limits, joint.limits should be used for all DOFs."""
        _, joints = expand_gimbal_joint(gimbal_joint)
        # gimbal_joint has default limits from JointType.REVOLUTE __post_init__
        for j in joints:
            assert j.limits is not None

    def test_custom_limits_per_dof(self, gimbal_joint_custom_axes: Joint) -> None:
        """Per-DOF composite_limits should be applied to each expanded joint."""
        _, joints = expand_gimbal_joint(gimbal_joint_custom_axes)
        assert joints[0].limits.lower == pytest.approx(-1.5)
        assert joints[0].limits.upper == pytest.approx(1.5)
        assert joints[0].limits.effort == pytest.approx(500.0)
        assert joints[1].limits.lower == pytest.approx(-0.5)
        assert joints[1].limits.upper == pytest.approx(0.5)
        assert joints[1].limits.effort == pytest.approx(300.0)
        assert joints[2].limits.lower == pytest.approx(-2.0)
        assert joints[2].limits.upper == pytest.approx(0.5)
        assert joints[2].limits.effort == pytest.approx(400.0)

    def test_joint_names(self, gimbal_joint: Joint) -> None:
        """Expanded joints should be named {original}_dof1, _dof2, _dof3."""
        _, joints = expand_gimbal_joint(gimbal_joint)
        assert joints[0].name == "shoulder_dof1"
        assert joints[1].name == "shoulder_dof2"
        assert joints[2].name == "shoulder_dof3"


# ---------------------------------------------------------------------------
# Universal joint expansion tests
# ---------------------------------------------------------------------------


class TestExpandUniversalJoint:
    """Tests for expand_universal_joint()."""

    def test_produces_one_intermediate_link(self, universal_joint: Joint) -> None:
        """Universal expansion should produce exactly 1 intermediate link."""
        links, _ = expand_universal_joint(universal_joint)
        assert len(links) == 1

    def test_produces_two_revolute_joints(self, universal_joint: Joint) -> None:
        """Universal expansion should produce exactly 2 revolute joints."""
        _, joints = expand_universal_joint(universal_joint)
        assert len(joints) == 2

    def test_all_joints_are_revolute(self, universal_joint: Joint) -> None:
        """All expanded joints should be of type REVOLUTE."""
        _, joints = expand_universal_joint(universal_joint)
        for j in joints:
            assert j.joint_type == JointType.REVOLUTE

    def test_intermediate_link_name(self, universal_joint: Joint) -> None:
        """Intermediate link should be named consistently."""
        links, _ = expand_universal_joint(universal_joint)
        assert links[0].name == "wrist_intermediate"

    def test_intermediate_link_mass(self, universal_joint: Joint) -> None:
        """Intermediate link should have the standard intermediate mass."""
        links, _ = expand_universal_joint(universal_joint)
        assert links[0].inertia.mass == INTERMEDIATE_LINK_MASS

    def test_intermediate_link_inertia_small(self, universal_joint: Joint) -> None:
        """Intermediate link should have negligible inertia."""
        links, _ = expand_universal_joint(universal_joint)
        assert links[0].inertia.ixx == pytest.approx(1e-6)
        assert links[0].inertia.iyy == pytest.approx(1e-6)
        assert links[0].inertia.izz == pytest.approx(1e-6)

    def test_parent_child_connectivity(self, universal_joint: Joint) -> None:
        """The chain should be: parent -> intermediate -> child."""
        links, joints = expand_universal_joint(universal_joint)
        # Joint 1: parent -> intermediate
        assert joints[0].parent == "forearm"
        assert joints[0].child == "wrist_intermediate"
        # Joint 2: intermediate -> child
        assert joints[1].parent == "wrist_intermediate"
        assert joints[1].child == "hand"

    def test_default_axes_xy(self, universal_joint: Joint) -> None:
        """Default axes should be X then Y."""
        _, joints = expand_universal_joint(universal_joint)
        assert joints[0].axis == (1, 0, 0)  # X
        assert joints[1].axis == (0, 1, 0)  # Y

    def test_custom_axes(self, universal_joint_custom_axes: Joint) -> None:
        """Custom composite_axes should be respected."""
        _, joints = expand_universal_joint(universal_joint_custom_axes)
        assert joints[0].axis == (0, 1, 0)
        assert joints[1].axis == (1, 0, 0)

    def test_origin_only_on_first_joint(self, universal_joint: Joint) -> None:
        """Only the first joint should carry the original origin."""
        _, joints = expand_universal_joint(universal_joint)
        assert joints[0].origin.xyz == (0.0, 0.0, -0.25)
        assert joints[1].origin.xyz == (0.0, 0.0, 0.0)

    def test_dynamics_propagated(self, universal_joint: Joint) -> None:
        """Dynamics should propagate to all expanded joints."""
        _, joints = expand_universal_joint(universal_joint)
        for j in joints:
            assert j.dynamics.damping == 0.5
            assert j.dynamics.friction == 0.05

    def test_limits_propagation_default(self, universal_joint: Joint) -> None:
        """When no composite_limits, joint.limits should be used for all DOFs."""
        _, joints = expand_universal_joint(universal_joint)
        for j in joints:
            assert j.limits is not None

    def test_custom_limits_per_dof(self, universal_joint_custom_axes: Joint) -> None:
        """Per-DOF composite_limits should be applied to each expanded joint."""
        _, joints = expand_universal_joint(universal_joint_custom_axes)
        assert joints[0].limits.lower == pytest.approx(-0.8)
        assert joints[0].limits.upper == pytest.approx(0.5)
        assert joints[0].limits.effort == pytest.approx(200.0)
        assert joints[1].limits.lower == pytest.approx(-0.3)
        assert joints[1].limits.upper == pytest.approx(0.3)
        assert joints[1].limits.effort == pytest.approx(150.0)

    def test_joint_names(self, universal_joint: Joint) -> None:
        """Expanded joints should be named {original}_dof1, _dof2."""
        _, joints = expand_universal_joint(universal_joint)
        assert joints[0].name == "wrist_dof1"
        assert joints[1].name == "wrist_dof2"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCompositeJointEdgeCases:
    """Edge-case tests for both expansion functions."""

    def test_gimbal_chain_is_contiguous(self, gimbal_joint: Joint) -> None:
        """Verify every child of joint N is the parent of joint N+1."""
        _, joints = expand_gimbal_joint(gimbal_joint)
        for i in range(len(joints) - 1):
            assert joints[i].child == joints[i + 1].parent

    def test_universal_chain_is_contiguous(self, universal_joint: Joint) -> None:
        """Verify the child of joint 0 is the parent of joint 1."""
        _, joints = expand_universal_joint(universal_joint)
        assert joints[0].child == joints[1].parent

    def test_gimbal_with_none_limits_uses_defaults(self) -> None:
        """A gimbal joint with limits=None should still produce valid limits."""
        joint = Joint(
            name="test_gimbal",
            joint_type=JointType.GIMBAL,
            parent="a",
            child="b",
        )
        _, joints = expand_gimbal_joint(joint)
        for j in joints:
            assert j.limits is not None
            assert j.limits.lower == pytest.approx(-math.pi)
            assert j.limits.upper == pytest.approx(math.pi)

    def test_universal_with_none_limits_uses_defaults(self) -> None:
        """A universal joint with limits=None should still produce valid limits."""
        joint = Joint(
            name="test_universal",
            joint_type=JointType.UNIVERSAL,
            parent="a",
            child="b",
        )
        _, joints = expand_universal_joint(joint)
        for j in joints:
            assert j.limits is not None

    def test_gimbal_preserves_rpy_on_first_joint(self) -> None:
        """The first joint in a gimbal chain should preserve the original RPY."""
        joint = Joint(
            name="test",
            joint_type=JointType.GIMBAL,
            parent="a",
            child="b",
            origin=Origin(xyz=(1.0, 2.0, 3.0), rpy=(0.1, 0.2, 0.3)),
        )
        _, joints = expand_gimbal_joint(joint)
        assert joints[0].origin.rpy == (0.1, 0.2, 0.3)
        assert joints[1].origin.rpy == (0.0, 0.0, 0.0)
        assert joints[2].origin.rpy == (0.0, 0.0, 0.0)

    def test_universal_preserves_rpy_on_first_joint(self) -> None:
        """The first joint in a universal chain should preserve the original RPY."""
        joint = Joint(
            name="test",
            joint_type=JointType.UNIVERSAL,
            parent="a",
            child="b",
            origin=Origin(xyz=(1.0, 2.0, 3.0), rpy=(0.4, 0.5, 0.6)),
        )
        _, joints = expand_universal_joint(joint)
        assert joints[0].origin.rpy == (0.4, 0.5, 0.6)
        assert joints[1].origin.rpy == (0.0, 0.0, 0.0)
