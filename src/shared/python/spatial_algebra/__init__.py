"""
Shared spatial algebra utilities.

This package provides a canonical implementation of Featherstone-style
spatial algebra that can be reused across physics engines.

Includes 6DOF positioning support for intuitive model placement.
"""

from .inertia import mcI, mci, transform_spatial_inertia
from .joints import (
    JOINT_AXIS_INDICES,
    S_PX,
    S_PY,
    S_PZ,
    S_RX,
    S_RY,
    S_RZ,
    jcalc,
)
from .pose6dof import (
    EntityPlacement,
    PlacementGroup,
    Pose6DOF,
    Transform6DOF,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    euler_to_rotation_matrix,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
    slerp,
)
from .spatial_vectors import (
    crf,
    crm,
    cross_force,
    cross_force_fast,
    cross_motion,
    cross_motion_axis,
    cross_motion_fast,
    skew,
    spatial_cross,
)
from .transforms import inv_xtrans, xlt, xrot, xtrans

__all__ = [
    # 6DOF Positioning
    "EntityPlacement",
    "PlacementGroup",
    "Pose6DOF",
    "Transform6DOF",
    # Rotation conversions
    "axis_angle_to_rotation_matrix",
    "euler_to_quaternion",
    "euler_to_rotation_matrix",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_to_euler",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_euler",
    "rotation_matrix_to_quaternion",
    "slerp",
    # Joint constants
    "JOINT_AXIS_INDICES",
    "S_PX",
    "S_PY",
    "S_PZ",
    "S_RX",
    "S_RY",
    "S_RZ",
    # Spatial operations
    "crf",
    "crm",
    "cross_force",
    "cross_force_fast",
    "cross_motion",
    "cross_motion_axis",
    "cross_motion_fast",
    "inv_xtrans",
    "jcalc",
    "mcI",
    "mci",
    "skew",
    "spatial_cross",
    "transform_spatial_inertia",
    "xlt",
    "xrot",
    "xtrans",
]
