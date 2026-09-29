"""Tests for composite joint expansion."""

from __future__ import annotations

from model_generation.core.composite_joints import (
    expand_gimbal_joint,
    expand_universal_joint,
)
from model_generation.core.types import (
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Origin,
)


def _gimbal(name: str = "g") -> Joint:
    return Joint(
        name=name,
        joint_type=JointType.GIMBAL,
        parent="P",
        child="C",
        origin=Origin(xyz=(1, 2, 3)),
        dynamics=JointDynamics(damping=0.42),
    )


def _universal(name: str = "u") -> Joint:
    return Joint(
        name=name,
        joint_type=JointType.UNIVERSAL,
        parent="P",
        child="C",
        origin=Origin(xyz=(1, 0, 0)),
    )


def test_expand_gimbal_creates_3_revolutes_and_2_links() -> None:
    links, joints = expand_gimbal_joint(_gimbal())
    assert len(links) == 2
    assert len(joints) == 3
    assert all(j.joint_type == JointType.REVOLUTE for j in joints)


def test_expand_gimbal_default_zyx_axes() -> None:
    _, joints = expand_gimbal_joint(_gimbal())
    assert joints[0].axis == (0, 0, 1)
    assert joints[1].axis == (0, 1, 0)
    assert joints[2].axis == (1, 0, 0)


def test_expand_gimbal_chain_parent_child() -> None:
    _, joints = expand_gimbal_joint(_gimbal("g"))
    # parent -> dof1 -> int_1 -> dof2 -> int_2 -> dof3 -> child
    assert joints[0].parent == "P"
    assert joints[0].child == "g_intermediate_1"
    assert joints[1].parent == "g_intermediate_1"
    assert joints[1].child == "g_intermediate_2"
    assert joints[2].parent == "g_intermediate_2"
    assert joints[2].child == "C"


def test_expand_gimbal_origin_only_on_first() -> None:
    _, joints = expand_gimbal_joint(_gimbal())
    assert joints[0].origin.xyz == (1, 2, 3)
    assert joints[1].origin.xyz == (0.0, 0.0, 0.0)
    assert joints[2].origin.xyz == (0.0, 0.0, 0.0)


def test_expand_gimbal_dynamics_propagated() -> None:
    _, joints = expand_gimbal_joint(_gimbal())
    assert all(j.dynamics.damping == 0.42 for j in joints)


def test_expand_gimbal_custom_axes_and_limits() -> None:
    j = _gimbal()
    j.composite_axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    j.composite_limits = [
        JointLimits(lower=-0.1, upper=0.1),
        JointLimits(lower=-0.2, upper=0.2),
        JointLimits(lower=-0.3, upper=0.3),
    ]
    _, joints = expand_gimbal_joint(j)
    assert joints[0].axis == (1, 0, 0)
    assert joints[1].limits.upper == 0.2


def test_expand_gimbal_intermediate_link_mass_nonzero_but_small() -> None:
    links, _ = expand_gimbal_joint(_gimbal())
    for link in links:
        assert 0 < link.inertia.mass < 1e-2


def test_expand_universal_creates_2_revolutes_and_1_link() -> None:
    links, joints = expand_universal_joint(_universal())
    assert len(links) == 1
    assert len(joints) == 2
    assert all(j.joint_type == JointType.REVOLUTE for j in joints)


def test_expand_universal_default_xy_axes() -> None:
    _, joints = expand_universal_joint(_universal())
    assert joints[0].axis == (1, 0, 0)
    assert joints[1].axis == (0, 1, 0)


def test_expand_universal_chain() -> None:
    _, joints = expand_universal_joint(_universal("u"))
    assert joints[0].parent == "P"
    assert joints[0].child == "u_intermediate"
    assert joints[1].parent == "u_intermediate"
    assert joints[1].child == "C"


def test_expand_universal_origin_only_on_first() -> None:
    _, joints = expand_universal_joint(_universal())
    assert joints[0].origin.xyz == (1, 0, 0)
    assert joints[1].origin.xyz == (0.0, 0.0, 0.0)


def test_expand_universal_custom_axes() -> None:
    j = _universal()
    j.composite_axes = [(0, 0, 1), (1, 0, 0)]
    _, joints = expand_universal_joint(j)
    assert joints[0].axis == (0, 0, 1)
    assert joints[1].axis == (1, 0, 0)
