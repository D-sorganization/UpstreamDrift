"""
Forward-dynamics motion matching solvers.

Part of issue #4568. Converts kinematic JointTrajectory into dynamically
consistent, torque-driven trajectories that forward-dynamics models track.
"""

from .base import (
    MotionMatchingSolver,
    MotionMatchingRequest,
    MotionMatchingResult,
    CostWeights,
    make_matching_solver,
    MatchingBackendType,
)

__all__ = [
    "MotionMatchingSolver",
    "MotionMatchingRequest",
    "MotionMatchingResult",
    "CostWeights",
    "make_matching_solver",
    "MatchingBackendType",
]