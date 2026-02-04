"""Locomotion module for bipedal and quadruped robots.

This module provides components for walking and balance:
- Gait state machines and phase management
- ZMP/CoP computation and tracking
- Footstep planning and execution
- Balance criteria evaluation

Example:
    >>> from src.robotics.locomotion import GaitStateMachine, ZMPComputer
    >>>
    >>> gait = GaitStateMachine(gait_type=GaitType.WALK)
    >>> zmp = ZMPComputer(engine)
    >>> zmp_pos = zmp.compute_zmp()
"""

from __future__ import annotations

from src.robotics.locomotion.footstep_planner import (
    Footstep,
    FootstepPlan,
    FootstepPlanner,
)
from src.robotics.locomotion.gait_state_machine import (
    GaitEvent,
    GaitState,
    GaitStateMachine,
)
from src.robotics.locomotion.gait_types import (
    GaitParameters,
    GaitPhase,
    GaitType,
    LegState,
    create_run_parameters,
    create_stand_parameters,
    create_walk_parameters,
)
from src.robotics.locomotion.zmp_computer import (
    ZMPComputer,
    ZMPResult,
)

__all__ = [
    # Gait Types
    "GaitType",
    "GaitPhase",
    "LegState",
    "GaitParameters",
    "create_walk_parameters",
    "create_run_parameters",
    "create_stand_parameters",
    # ZMP
    "ZMPComputer",
    "ZMPResult",
    # State Machine
    "GaitStateMachine",
    "GaitState",
    "GaitEvent",
    # Footsteps
    "Footstep",
    "FootstepPlan",
    "FootstepPlanner",
]
