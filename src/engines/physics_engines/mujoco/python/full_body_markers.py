"""Marker kinematics and pose inverse kinematics on the MuJoCo full-body model (deprecated).

This module is deprecated; import from
`src.engines.physics_engines.mujoco.python.full_body_ik` or
`src.shared.python.motion_matching.full_body_ik` instead.
"""

from __future__ import annotations

import warnings

from src.engines.physics_engines.mujoco.python.full_body_ik import (
    Array,
    Attachment,
    FullBodyMarkerKinematics,
)
from src.shared.python.motion_matching.full_body_ik import (
    PoseFit,
    SolvePoseOptions,
    SolveTrajectoryOptions,
    _rotation_error,
    continuous_branches,
)

warnings.warn(
    "src.engines.physics_engines.mujoco.python.full_body_markers is deprecated; "
    "import from src.engines.physics_engines.mujoco.python.full_body_ik or "
    "src.shared.python.motion_matching.full_body_ik instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Array",
    "Attachment",
    "FullBodyMarkerKinematics",
    "PoseFit",
    "SolvePoseOptions",
    "SolveTrajectoryOptions",
    "_rotation_error",
    "continuous_branches",
]
