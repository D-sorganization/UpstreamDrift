"""UpstreamDrift Robotics Module.

By Robots, For Robots.

This module provides comprehensive robotics capabilities including:
- Contact dynamics and multi-contact management
- Force/torque sensing with realistic noise models
- Whole-body control with hierarchical task prioritization
- Bipedal locomotion with ZMP, DCM, and CPG-based gaits
- Analysis metrics for locomotion and manipulation

Design Principles:
- Engine-agnostic: Works with any physics engine via protocols
- Design by Contract: All functions have documented pre/postconditions
- Test-driven: Comprehensive test coverage required
- DRY, orthogonal, reversible, reusable code

Example:
    >>> from src.robotics import ContactManager, WholeBodyController
    >>> from src.robotics.core.protocols import HumanoidCapable
    >>>
    >>> # Create controller for any compatible engine
    >>> contact_mgr = ContactManager(engine)
    >>> wbc = WholeBodyController(engine)
    >>>
    >>> # Detect contacts and solve whole-body control
    >>> contacts = contact_mgr.detect_contacts()
    >>> torques = wbc.solve(tasks, contacts)
"""

from __future__ import annotations

# Contact dynamics
from src.robotics.contact import (
    ContactManager,
    FrictionCone,
    check_force_closure,
    compute_friction_cone_constraint,
    compute_grasp_matrix,
    compute_grasp_quality,
    linearize_friction_cone,
)

# Whole-body control
from src.robotics.control import (
    QPProblem,
    QPSolution,
    QPSolver,
    Task,
    TaskType,
    WBCConfig,
    WBCSolution,
    WholeBodyController,
    create_com_task,
    create_ee_task,
    create_posture_task,
)

# Core exceptions
from src.robotics.core.exceptions import (
    ContactError,
    ControlError,
    LocomotionError,
    RoboticsError,
    SolverError,
)

# Core protocols
from src.robotics.core.protocols import (
    ContactCapable,
    HumanoidCapable,
    ManipulationCapable,
    RoboticsCapable,
)

# Core types
from src.robotics.core.types import (
    ContactState,
    ContactType,
    ControlMode,
    FrictionConeType,
    GaitPhase,
    SupportState,
    TaskPriority,
)

# Locomotion
from src.robotics.locomotion import (
    Footstep,
    FootstepPlan,
    FootstepPlanner,
    GaitEvent,
    GaitParameters,
    GaitState,
    GaitStateMachine,
    GaitType,
    LegState,
    ZMPComputer,
    ZMPResult,
    create_run_parameters,
    create_stand_parameters,
    create_walk_parameters,
)

# Sensing
from src.robotics.sensing import (
    BrownianNoise,
    CompositeNoise,
    ForceTorqueSensor,
    ForceTorqueSensorConfig,
    GaussianNoise,
    IMUSensor,
    IMUSensorConfig,
    NoiseModel,
    QuantizationNoise,
)

__all__ = [
    # Exceptions
    "RoboticsError",
    "ContactError",
    "ControlError",
    "LocomotionError",
    "SolverError",
    # Core Types
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
    # Contact
    "ContactManager",
    "FrictionCone",
    "linearize_friction_cone",
    "compute_friction_cone_constraint",
    "compute_grasp_matrix",
    "check_force_closure",
    "compute_grasp_quality",
    # Sensing
    "ForceTorqueSensor",
    "ForceTorqueSensorConfig",
    "IMUSensor",
    "IMUSensorConfig",
    "NoiseModel",
    "GaussianNoise",
    "BrownianNoise",
    "QuantizationNoise",
    "CompositeNoise",
    # Control
    "WholeBodyController",
    "WBCConfig",
    "WBCSolution",
    "Task",
    "TaskType",
    "create_com_task",
    "create_posture_task",
    "create_ee_task",
    "QPSolver",
    "QPProblem",
    "QPSolution",
    # Locomotion
    "GaitType",
    "GaitParameters",
    "LegState",
    "create_walk_parameters",
    "create_run_parameters",
    "create_stand_parameters",
    "ZMPComputer",
    "ZMPResult",
    "GaitStateMachine",
    "GaitState",
    "GaitEvent",
    "Footstep",
    "FootstepPlan",
    "FootstepPlanner",
]

__version__ = "0.1.0"
