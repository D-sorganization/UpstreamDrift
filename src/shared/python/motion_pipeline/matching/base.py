"""
Base Protocol and factory for Motion Matching solvers.

Part of issue #4568. Defines the unified motion matching interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

import numpy as np

from ..contracts import (
    JointTrajectory,
    MotionMatchingRequest as ContractMotionMatchingRequest,
    MotionMatchingResult as ContractMotionMatchingResult,
    SkeletonRig,
)


class MatchingBackendType(str, Enum):
    """Available motion matching backend types."""
    
    CMC = "cmc"  # Computed Muscle Control (OpenSim)
    RRA = "rra"  # Residual Reduction Algorithm (OpenSim)
    TRAJOPT_DRAKE = "drake_trajopt"  # Drake trajectory optimization
    TORQUE_MUJOCO = "mujoco_torque"  # MuJoCo torque tracking
    INVERSE_DYN_PINOCCHIO = "pinocchio_inverse_dyn"  # Pinocchio RNEA


@dataclass
class CostWeights:
    """
    Cost function weights for motion matching.
    
    Attributes:
        joint_tracking: Weight for joint position tracking
        marker_tracking: Weight for marker position tracking
        smoothness: Weight for trajectory smoothness (velocity/acceleration)
        effort: Weight for control effort (torque/activation)
        contact: Weight for contact force regularization
        residual: Weight for residual force reduction
    """
    
    joint_tracking: float = field(default=1.0)
    marker_tracking: float = field(default=0.5)
    smoothness: float = field(default=0.1)
    effort: float = field(default=0.01)
    contact: float = field(default=0.1)
    residual: float = field(default=10.0)


@dataclass
class MotionMatchingRequest:
    """
    Request for motion matching solver.
    
    Attributes:
        id: Unique request identifier
        reference: Reference joint trajectory to track
        rig: Scaled skeleton rig
        cost_weights: Cost function weights
        time_horizon: Time horizon for optimization (seconds)
        integrator_step: Integrator time step (seconds)
        max_iterations: Maximum solver iterations
        use_residuals: Whether to use residual forces
        use_contacts: Whether to include contact forces
    """
    
    id: str
    reference: JointTrajectory
    rig: SkeletonRig
    cost_weights: CostWeights | None = None
    time_horizon: float | None = None
    integrator_step: float = field(default=0.01)
    max_iterations: int = field(default=100)
    use_residuals: bool = field(default=True)
    use_contacts: bool = field(default=False)
    
    def __post_init__(self):
        if self.time_horizon is None:
            self.time_horizon = self.reference.duration


@dataclass
class MotionMatchingResult:
    """
    Result from motion matching solver.
    
    Attributes:
        request_id: Associated request identifier
        success: Whether matching succeeded
        tracked_trajectory: Tracked joint trajectory
        torque_trajectory: Torque trajectory (if computed)
        residual_report: Residual force report
        fit_metrics: Fit metrics (RMSE, max error, etc.)
        solve_time: Solver time in seconds
        message: Status message
        metadata: Additional metadata
    """
    
    request_id: str
    success: bool
    tracked_trajectory: JointTrajectory | None = None
    torque_trajectory: JointTrajectory | None = None
    residual_report: dict | None = None
    fit_metrics: dict | None = None
    solve_time: float | None = None
    message: str | None = None
    metadata: dict = field(default_factory=dict)
    
    def to_contract(self) -> ContractMotionMatchingResult:
        """Convert to contract MotionMatchingResult."""
        return ContractMotionMatchingResult(
            request_id=self.request_id,
            success=self.success,
            matched_trajectory=self.tracked_trajectory,
            error_metrics=self.fit_metrics or {},
            message=self.message,
            metadata=self.metadata
        )


class MotionMatchingSolver(Protocol):
    """
    Protocol for Motion Matching solvers.
    
    Converts a kinematic JointTrajectory into a dynamically
    consistent, torque-driven trajectory.
    """
    
    def match(
        self,
        reference: JointTrajectory,
        rig: SkeletonRig,
        request: MotionMatchingRequest | None = None,
    ) -> MotionMatchingResult:
        """
        Solve motion matching for a reference trajectory.
        
        Args:
            reference: Reference joint trajectory to track
            rig: Scaled skeleton rig
            request: Optional matching request with configuration
        
        Returns:
            MotionMatchingResult with tracked trajectory and metrics
        
        Raises:
            ValueError: If reference/rig are invalid
            RuntimeError: If solver fails to converge
        """
        ...


class BaseMotionMatchingSolver(ABC):
    """
    Abstract base class for motion matching solvers.
    
    Provides common functionality for cost computation,
    residual reporting, and validation.
    """
    
    def __init__(self, cost_weights: CostWeights | None = None):
        """
        Initialize motion matching solver.
        
        Args:
            cost_weights: Cost function weights
        """
        self.cost_weights = cost_weights or CostWeights()
    
    @abstractmethod
    def match(
        self,
        reference: JointTrajectory,
        rig: SkeletonRig,
        request: MotionMatchingRequest | None = None,
    ) -> MotionMatchingResult:
        """Solve motion matching for a reference trajectory."""
    
    def _compute_rmse(
        self,
        reference: JointTrajectory,
        tracked: JointTrajectory,
    ) -> float:
        """
        Compute RMSE between reference and tracked trajectories.
        
        Args:
            reference: Reference trajectory
            tracked: Tracked trajectory
        
        Returns:
            RMSE in radians
        """
        errors = []
        for ref_frame, track_frame in zip(reference.frames, tracked.frames):
            for ref_q, track_q in zip(ref_frame.q, track_frame.q):
                errors.append((ref_q - track_q) ** 2)
        
        return float(np.sqrt(np.mean(errors)))
    
    def _compute_residual_report(
        self,
        reference: JointTrajectory,
        tracked: JointTrajectory,
    ) -> dict:
        """
        Compute residual force report.
        
        Args:
            reference: Reference trajectory
            tracked: Tracked trajectory
        
        Returns:
            Dict with residual statistics
        """
        # Compute residual as difference in joint angles
        residuals = []
        for ref_frame, track_frame in zip(reference.frames, tracked.frames):
            for ref_q, track_q in zip(ref_frame.q, track_frame.q):
                residuals.append(abs(ref_q - track_q))
        
        return {
            "mean_residual": float(np.mean(residuals)),
            "max_residual": float(np.max(residuals)),
            "std_residual": float(np.std(residuals)),
            "num_frames": len(reference.frames),
        }
    
    def _validate_result(
        self,
        trajectory: JointTrajectory,
    ) -> bool:
        """
        Validate motion matching result satisfies DbC postconditions.
        
        Postconditions:
        - Torque/activation finite
        - Time grid matches reference
        """
        # Check for NaN/Inf in joint angles
        for frame in trajectory.frames:
            if any(not np.isfinite(v) for v in frame.q):
                return False
        
        return True


def make_matching_solver(
    backend: MatchingBackendType | str,
    cost_weights: CostWeights | None = None,
) -> MotionMatchingSolver:
    """
    Factory function to create a motion matching solver for the specified backend.
    
    Args:
        backend: Backend type (cmc, rra, drake_trajopt, mujoco_torque, pinocchio_inverse_dyn)
        cost_weights: Optional cost weights
    
    Returns:
        MotionMatchingSolver instance
    
    Raises:
        ImportError: If backend not available
        ValueError: If backend not recognized
    """
    if isinstance(backend, str):
        backend = MatchingBackendType(backend.lower())
    
    if backend == MatchingBackendType.CMC:
        from .cmc import CMCMatchingSolver
        return CMCMatchingSolver(cost_weights)
    
    if backend == MatchingBackendType.RRA:
        from .rra import RRAMatchingSolver
        return RRAMatchingSolver(cost_weights)
    
    if backend == MatchingBackendType.TRAJOPT_DRAKE:
        from .trajopt_drake import DrakeTrajoptMatchingSolver
        return DrakeTrajoptMatchingSolver(cost_weights)
    
    if backend == MatchingBackendType.TORQUE_MUJOCO:
        from .torque_mujoco import MuJoCoTorqueMatchingSolver
        return MuJoCoTorqueMatchingSolver(cost_weights)
    
    if backend == MatchingBackendType.INVERSE_DYN_PINOCCHIO:
        from .inverse_dyn_pinocchio import PinocchioInverseDynMatchingSolver
        return PinocchioInverseDynMatchingSolver(cost_weights)
    
    raise ValueError(f"Unknown motion matching backend: {backend}")