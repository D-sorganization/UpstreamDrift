"""Shared constants for URDF import/export functionality.

Joint type mapping tables between MuJoCo MJCF and URDF formats.
"""

from __future__ import annotations

import mujoco

# Joint type mappings between MuJoCo and URDF
MJCF_TO_URDF_JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_HINGE: "revolute",
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
    mujoco.mjtJoint.mjJNT_FREE: "floating",  # Not standard URDF, handled specially
    mujoco.mjtJoint.mjJNT_BALL: "spherical",  # Not standard URDF, approximated
}

URDF_TO_MJCF_JOINT_TYPES = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
    "continuous": mujoco.mjtJoint.mjJNT_HINGE,  # Continuous = unlimited revolute
    "fixed": None,  # Fixed joints are handled differently in MuJoCo
    "floating": mujoco.mjtJoint.mjJNT_FREE,
}
