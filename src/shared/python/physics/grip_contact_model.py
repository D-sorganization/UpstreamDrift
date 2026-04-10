"""Contact-Based Grip Model Module.

Guideline K2 Implementation: Contact-Based Grip Model (MuJoCo).

Provides contact mechanics modeling for hand-club interface including:
- Friction cone constraints (static/dynamic coefficients)
- Normal force distribution across contact points
- Slip detection and magnitude tracking
- Grip pressure visualization data

The grip model replaces rigid constraints with contact pairs,
enabling realistic force transmission and slip analysis.
"""

from __future__ import annotations

from src.shared.python.physics._contact_types import (
    DEFAULT_DYNAMIC_FRICTION,
    DEFAULT_STATIC_FRICTION,
    SLIP_VELOCITY_THRESHOLD,
    ContactPoint,
    ContactState,
    GripContactState,
    GripContactTimestep,
    GripParameters,
    PressureVisualizationData,
)
from src.shared.python.physics._friction_laws import (
    check_friction_cone,
    classify_contact_state,
    compute_slip_direction,
    decompose_contact_force,
)
from src.shared.python.physics._grip_exporter import (
    GripContactExporter,
    create_mujoco_grip_contacts,
)
from src.shared.python.physics._grip_forces import (
    compute_center_of_pressure,
    compute_grip_torque,
    compute_pressure_visualization,
)
from src.shared.python.physics._grip_model import GripContactModel

__all__ = [
    "DEFAULT_DYNAMIC_FRICTION",
    "DEFAULT_STATIC_FRICTION",
    "SLIP_VELOCITY_THRESHOLD",
    "ContactPoint",
    "ContactState",
    "GripContactExporter",
    "GripContactModel",
    "GripContactState",
    "GripContactTimestep",
    "GripParameters",
    "PressureVisualizationData",
    "check_friction_cone",
    "classify_contact_state",
    "compute_center_of_pressure",
    "compute_grip_torque",
    "compute_pressure_visualization",
    "compute_slip_direction",
    "create_mujoco_grip_contacts",
    "decompose_contact_force",
]
