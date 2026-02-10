"""Backward compatibility shim - module moved to physics.grip_contact_model."""

import sys as _sys

from .physics import grip_contact_model as _real_module  # noqa: E402
from .physics.grip_contact_model import (  # noqa: F401
    DEFAULT_DYNAMIC_FRICTION,
    DEFAULT_STATIC_FRICTION,
    SLIP_VELOCITY_THRESHOLD,
    ContactPoint,
    ContactState,
    GripContactExporter,
    GripContactModel,
    GripContactState,
    GripContactTimestep,
    GripParameters,
    PressureVisualizationData,
    check_friction_cone,
    classify_contact_state,
    compute_center_of_pressure,
    compute_grip_torque,
    compute_pressure_visualization,
    compute_slip_direction,
    create_mujoco_grip_contacts,
    decompose_contact_force,
    logger,
)

_sys.modules[__name__] = _real_module
