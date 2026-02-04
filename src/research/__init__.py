"""Advanced Research Capabilities for UpstreamDrift.

This module provides advanced research tools:
- Model Predictive Control (MPC) frameworks
- Differentiable Physics simulation
- Deformable Object simulation
- Multi-Robot Coordination

Phase 5 of the Robotics Expansion Proposal.
"""

from __future__ import annotations

from src.research.mpc import (
    CentroidalMPC,
    CostFunction,
    ModelPredictiveController,
    MPCResult,
    WholeBodyMPC,
)
from src.research.differentiable import (
    ContactDifferentiableEngine,
    DifferentiableEngine,
)
from src.research.deformable import (
    Cable,
    Cloth,
    DeformableObject,
    MaterialProperties,
    SoftBody,
)
from src.research.multi_robot import (
    CooperativeManipulation,
    FormationConfig,
    FormationController,
    MultiRobotSystem,
    Task,
    TaskCoordinator,
)

__all__ = [
    # MPC
    "ModelPredictiveController",
    "CostFunction",
    "MPCResult",
    "CentroidalMPC",
    "WholeBodyMPC",
    # Differentiable
    "DifferentiableEngine",
    "ContactDifferentiableEngine",
    # Deformable
    "DeformableObject",
    "SoftBody",
    "Cable",
    "Cloth",
    "MaterialProperties",
    # Multi-Robot
    "MultiRobotSystem",
    "FormationController",
    "FormationConfig",
    "CooperativeManipulation",
    "Task",
    "TaskCoordinator",
]
