"""Core robotics abstractions and protocols.

This module defines the foundational types, protocols, and exceptions
used throughout the robotics module.
"""

from __future__ import annotations

from src.robotics.core.exceptions import (
    ContactError,
    ControlError,
    LocomotionError,
    RoboticsError,
    SolverError,
)
from src.robotics.core.protocols import (
    ContactCapable,
    HumanoidCapable,
    ManipulationCapable,
    RoboticsCapable,
)
from src.robotics.core.types import (
    ContactState,
    ContactType,
    ControlMode,
    FrictionConeType,
    GaitPhase,
    SupportState,
    TaskPriority,
)

__all__ = [
    # Exceptions
    "RoboticsError",
    "ContactError",
    "ControlError",
    "LocomotionError",
    "SolverError",
    # Types
    "ContactState",
    "ContactType",
    "FrictionConeType",
    "TaskPriority",
    "ControlMode",
    "GaitPhase",
    "SupportState",
    # Protocols
    "RoboticsCapable",
    "ContactCapable",
    "HumanoidCapable",
    "ManipulationCapable",
]
