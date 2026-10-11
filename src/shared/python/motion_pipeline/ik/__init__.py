"""
Unified Inverse Kinematics solver interface.

Part of issue #4566. Provides a single IK API that takes a MarkerTrajectory
(or KeypointSequence) + scaled SkeletonRig and returns a JointTrajectory.
Pluggable across MuJoCo, OpenSim, Drake, and Pinocchio backends.
"""

from .base import InverseKinematicsSolver, MarkerWeights, make_ik_solver, IKBackendType
from .opensim_backend import OpenSimIKBackend, OpenSimIKSolver

__all__ = [
    "InverseKinematicsSolver",
    "MarkerWeights",
    "make_ik_solver",
    "IKBackendType",
    "OpenSimIKBackend",
    "OpenSimIKSolver",
]
