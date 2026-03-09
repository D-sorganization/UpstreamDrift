"""
Shared composite joint expansion utilities.

This module provides standalone functions for expanding composite URDF joints
(gimbal, universal) into sequences of revolute joints with intermediate links.
These functions are the single source of truth for composite joint expansion
logic, used by both ``model_generation.builders.urdf_writer.URDFWriter`` and
any other callers that need to decompose multi-DOF joints into URDF-compatible
single-DOF revolute chains.

Design rationale:
    URDF does not natively support multi-DOF joints (gimbal/spherical,
    universal).  The standard workaround is to decompose them into a chain
    of single-axis revolute joints connected by massless intermediate links.
    This logic was previously duplicated inside ``URDFWriter`` methods; it is
    now extracted here so that parametric builders, converters, and the
    humanoid character builder can all share the same implementation.
"""

from __future__ import annotations

import logging

from model_generation.core.constants import INTERMEDIATE_LINK_MASS
from model_generation.core.types import (
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
    Origin,
)

logger = logging.getLogger(__name__)

# Default Euler-angle axis sequences
_GIMBAL_DEFAULT_AXES: list[tuple[float, float, float]] = [
    (0, 0, 1),  # Z
    (0, 1, 0),  # Y
    (1, 0, 0),  # X
]

_UNIVERSAL_DEFAULT_AXES: list[tuple[float, float, float]] = [
    (1, 0, 0),  # X
    (0, 1, 0),  # Y
]


def _make_intermediate_link(name: str) -> Link:
    """Create a near-massless intermediate link for composite joint chains."""
    return Link(
        name=name,
        inertia=Inertia(
            ixx=1e-6,
            iyy=1e-6,
            izz=1e-6,
            mass=INTERMEDIATE_LINK_MASS,
        ),
    )


def expand_gimbal_joint(joint: Joint) -> tuple[list[Link], list[Joint]]:
    """Expand a gimbal (3-DOF) joint into 3 revolute joints.

    A gimbal joint is decomposed into a Z-Y-X Euler-angle chain of three
    revolute joints connected by two near-massless intermediate links.

    Args:
        joint: A :class:`Joint` with ``joint_type == JointType.GIMBAL``.

    Returns:
        A tuple of ``(intermediate_links, revolute_joints)`` where
        ``intermediate_links`` contains 2 links and ``revolute_joints``
        contains 3 revolute joints forming the chain::

            parent -> dof1 -> intermediate_1 -> dof2 -> intermediate_2 -> dof3 -> child
    """
    axes = joint.composite_axes or list(_GIMBAL_DEFAULT_AXES)
    limits = joint.composite_limits or [joint.limits] * 3

    intermediate_links: list[Link] = []
    revolute_joints: list[Joint] = []

    # Create 2 intermediate links
    for i in range(2):
        link_name = f"{joint.name}_intermediate_{i + 1}"
        intermediate_links.append(_make_intermediate_link(link_name))

    # Build the parent/child chain
    parents = [
        joint.parent,
        f"{joint.name}_intermediate_1",
        f"{joint.name}_intermediate_2",
    ]
    children = [
        f"{joint.name}_intermediate_1",
        f"{joint.name}_intermediate_2",
        joint.child,
    ]

    for i in range(3):
        revolute_joints.append(
            Joint(
                name=f"{joint.name}_dof{i + 1}",
                joint_type=JointType.REVOLUTE,
                parent=parents[i],
                child=children[i],
                origin=joint.origin if i == 0 else Origin(),
                axis=axes[i] if i < len(axes) else (0, 0, 1),
                limits=limits[i] if limits and i < len(limits) else JointLimits(),
                dynamics=joint.dynamics,
            )
        )

    logger.debug(
        "Expanded gimbal joint '%s' into 3 revolute joints with 2 intermediate links",
        joint.name,
    )
    return intermediate_links, revolute_joints


def expand_universal_joint(joint: Joint) -> tuple[list[Link], list[Joint]]:
    """Expand a universal (2-DOF) joint into 2 revolute joints.

    A universal joint is decomposed into two perpendicular revolute joints
    connected by one near-massless intermediate link.

    Args:
        joint: A :class:`Joint` with ``joint_type == JointType.UNIVERSAL``.

    Returns:
        A tuple of ``(intermediate_links, revolute_joints)`` where
        ``intermediate_links`` contains 1 link and ``revolute_joints``
        contains 2 revolute joints forming the chain::

            parent -> dof1 -> intermediate -> dof2 -> child
    """
    axes = joint.composite_axes or list(_UNIVERSAL_DEFAULT_AXES)
    limits = joint.composite_limits or [joint.limits] * 2

    intermediate_links: list[Link] = []
    revolute_joints: list[Joint] = []

    # Create one intermediate link
    link_name = f"{joint.name}_intermediate"
    intermediate_links.append(_make_intermediate_link(link_name))

    # Create 2 revolute joints
    for i in range(2):
        parent = joint.parent if i == 0 else link_name
        child = link_name if i == 0 else joint.child

        revolute_joints.append(
            Joint(
                name=f"{joint.name}_dof{i + 1}",
                joint_type=JointType.REVOLUTE,
                parent=parent,
                child=child,
                origin=joint.origin if i == 0 else Origin(),
                axis=axes[i] if i < len(axes) else (0, 0, 1),
                limits=limits[i] if limits and i < len(limits) else JointLimits(),
                dynamics=joint.dynamics,
            )
        )

    logger.debug(
        "Expanded universal joint '%s' into 2 revolute joints with 1 intermediate link",
        joint.name,
    )
    return intermediate_links, revolute_joints
