"""Ground-supported forward dynamics of the MuJoCo full-body model (deprecated).

This module is deprecated; import from
`src.shared.python.motion_matching.full_body_forward_dynamics` instead.
"""

from __future__ import annotations

import warnings

from src.shared.python.motion_matching.full_body_forward_dynamics import (
    MIN_SINGULAR_VALUE,
    ROOT_COORDINATES,
    Array,
    ComputedTorqueGains,
    Controller,
    FullBodySimulator,
    SimulationRecord,
    _balance_acceleration,
    _check_gains,
    _computed_torque,
    _distance_outside,
    _planted_com_jacobian,
    _planted_coupling,
    _root_regulation_acceleration,
    hold_pose_controller,
    joint_natural_frequencies,
    preload_feet,
    reference_zmp,
    tracking_controller,
)

warnings.warn(
    "src.engines.physics_engines.mujoco.python.full_body_simulation is deprecated; "
    "import from src.shared.python.motion_matching.full_body_forward_dynamics instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "MIN_SINGULAR_VALUE",
    "ROOT_COORDINATES",
    "Array",
    "ComputedTorqueGains",
    "Controller",
    "FullBodySimulator",
    "SimulationRecord",
    "_balance_acceleration",
    "_check_gains",
    "_computed_torque",
    "_distance_outside",
    "_planted_com_jacobian",
    "_planted_coupling",
    "_root_regulation_acceleration",
    "hold_pose_controller",
    "joint_natural_frequencies",
    "preload_feet",
    "reference_zmp",
    "tracking_controller",
]
