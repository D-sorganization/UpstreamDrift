"""
Forward-dynamics motion matching solvers.

Part of issue #4568. Converts kinematic JointTrajectory into dynamically
consistent, torque-driven trajectories that forward-dynamics models track.
"""

from .base import (
    CostWeights,
    MatchingBackendType,
    MotionMatchingRequest,
    MotionMatchingResult,
    MotionMatchingSolver,
    make_matching_solver,
)
from .contact import (
    ContactModel,
    FlatGroundContact,
    NoContactModel,
    infer_contact_phases,
)
from .costs import (
    composite_cost,
    effort_cost,
    joint_tracking_cost,
    marker_tracking_cost,
    residual_cost,
    smoothness_cost,
)

__all__ = [
    "MotionMatchingSolver",
    "MotionMatchingRequest",
    "MotionMatchingResult",
    "CostWeights",
    "make_matching_solver",
    "MatchingBackendType",
    # costs
    "joint_tracking_cost",
    "marker_tracking_cost",
    "smoothness_cost",
    "effort_cost",
    "residual_cost",
    "composite_cost",
    # contact
    "ContactModel",
    "FlatGroundContact",
    "NoContactModel",
    "infer_contact_phases",
]
