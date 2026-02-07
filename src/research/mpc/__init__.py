"""Model Predictive Control (MPC) framework.

This module provides MPC implementations for robot control:
- Generic nonlinear MPC
- Centroidal dynamics MPC for locomotion
- Whole-body MPC for manipulation
"""

from __future__ import annotations

from src.research.mpc.controller import (
    CostFunction,
    ModelPredictiveController,
    MPCResult,
)
from src.research.mpc.specialized import (
    CentroidalMPC,
    WholeBodyMPC,
)

__all__ = [
    "ModelPredictiveController",
    "CostFunction",
    "MPCResult",
    "CentroidalMPC",
    "WholeBodyMPC",
]
