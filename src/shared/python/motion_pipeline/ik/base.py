"""
Base Protocol and factory for Inverse Kinematics solvers.

Part of issue #4566. Defines the unified IK solver interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

import numpy as np

from ..contracts import (
    JointTrajectory,
    MarkerTrajectory,
    SkeletonRig,
)


class IKBackendType(str, Enum):
    """Available IK backend types."""

    MUJOCO = "mujoco"
    OPENSIM = "opensim"
    DRAKE = "drake"
    PINOCCHIO = "pinocchio"
    GEOMETRIC = "geometric"  # Fallback geometric solver


@dataclass
class MarkerWeights:
    """
    Per-marker weights for IK solving.

    Attributes:
        default_weight: Default weight for all markers
        marker_weights: Per-marker weight overrides
    """

    default_weight: float = field(default=1.0)
    marker_weights: dict[str, float] = field(default_factory=dict)

    def get_weight(self, marker_name: str) -> float:
        """Get weight for a specific marker."""
        return self.marker_weights.get(marker_name, self.default_weight)


@dataclass
class IKConfig:
    """
    Configuration for IK solving.

    Attributes:
        max_iterations: Maximum solver iterations
        tolerance: Convergence tolerance (radians)
        use_orientation: Whether to use orientation constraints
        regularization: Regularization weight for joint limits
    """

    max_iterations: int = field(default=100)
    tolerance: float = field(default=1e-6)
    use_orientation: bool = field(default=True)
    regularization: float = field(default=0.01)


class InverseKinematicsSolver(Protocol):
    """
    Protocol for Inverse Kinematics solvers.

    Takes a MarkerTrajectory (or KeypointSequence) + scaled SkeletonRig
    and returns a JointTrajectory.
    """

    def solve(
        self,
        markers: MarkerTrajectory,
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
        config: IKConfig | None = None,
    ) -> JointTrajectory:
        """
        Solve inverse kinematics for a marker trajectory.

        Args:
            markers: Input marker trajectory
            rig: Scaled skeleton rig
            weights: Optional per-marker weights
            config: Optional solver configuration

        Returns:
            JointTrajectory with solved joint angles

        Raises:
            ValueError: If markers/rig are invalid
            RuntimeError: If solver fails to converge
        """
        ...

    def solve_frame(
        self,
        markers: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
    ) -> list[float]:
        """
        Solve IK for a single frame.

        Args:
            markers: Dict mapping marker names to (x, y, z) positions
            rig: Scaled skeleton rig
            weights: Optional per-marker weights

        Returns:
            List of joint angles (q) in radians
        """
        ...


class BaseIKSolver(ABC):
    """
    Abstract base class for IK solvers.

    Provides common functionality for marker weighting,
    joint limit clamping, and validation.
    """

    def __init__(self, config: IKConfig | None = None):
        """
        Initialize IK solver.

        Args:
            config: Solver configuration
        """
        self.config = config or IKConfig()

    @abstractmethod
    def solve(
        self,
        markers: MarkerTrajectory,
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
        config: IKConfig | None = None,
    ) -> JointTrajectory:
        """Solve IK for a marker trajectory."""

    @abstractmethod
    def solve_frame(
        self,
        markers: dict[str, tuple[float, float, float]],
        rig: SkeletonRig,
        weights: MarkerWeights | None = None,
    ) -> list[float]:
        """Solve IK for a single frame."""

    def _apply_weights(
        self,
        marker_names: list[str],
        weights: MarkerWeights | None,
    ) -> list[float]:
        """Apply marker weights."""
        if weights is None:
            return [1.0] * len(marker_names)
        return [weights.get_weight(name) for name in marker_names]

    def _clamp_to_limits(
        self,
        q: list[float],
        rig: SkeletonRig,
    ) -> list[float]:
        """Clamp joint angles to rig limits."""
        clamped = []
        dof_idx = 0
        for joint in rig.joints.values():
            num_dofs = len(joint.axes)
            if num_dofs == 0:
                continue

            for i in range(num_dofs):
                if dof_idx >= len(q):
                    break
                angle = q[dof_idx]
                if joint.limits and i < len(joint.limits):
                    limit = joint.limits[i]
                    if limit.lower is not None:
                        angle = max(angle, limit.lower)
                    if limit.upper is not None:
                        angle = min(angle, limit.upper)
                clamped.append(angle)
                dof_idx += 1

        # If there are any remaining elements in q (e.g. root transform), keep them as is
        while dof_idx < len(q):
            clamped.append(q[dof_idx])
            dof_idx += 1

        return clamped

    def _validate_result(
        self,
        q: list[float],
        rig: SkeletonRig,
    ) -> bool:
        """
        Validate IK result satisfies DbC postconditions.

        Postconditions:
        - All joint angles within limits
        - No NaN or infinite values
        """
        # Check for NaN/Inf
        if any(not np.isfinite(v) for v in q):
            return False

        # Check limits
        clamped = self._clamp_to_limits(q, rig)
        return np.allclose(q, clamped, atol=1e-6)


def make_ik_solver(
    backend: IKBackendType | str,
    config: IKConfig | None = None,
) -> InverseKinematicsSolver:
    """
    Factory function to create an IK solver for the specified backend.

    Args:
        backend: Backend type (mujoco, opensim, drake, pinocchio, geometric)
        config: Optional solver configuration

    Returns:
        InverseKinematicsSolver instance

    Raises:
        ImportError: If backend not available
        ValueError: If backend not recognized
    """
    if isinstance(backend, str):
        backend = IKBackendType(backend.lower())

    if backend == IKBackendType.MUJOCO:
        from .mujoco_backend import MuJoCoIKSolver

        return MuJoCoIKSolver(config)

    if backend == IKBackendType.OPENSIM:
        from .opensim_backend import OpenSimIKSolver

        return OpenSimIKSolver(config)

    if backend == IKBackendType.DRAKE:
        from .drake_backend import DrakeIKSolver

        return DrakeIKSolver(config)

    if backend == IKBackendType.PINOCCHIO:
        from .pinocchio_backend import PinocchioIKSolver

        return PinocchioIKSolver(config)

    if backend == IKBackendType.GEOMETRIC:
        from .geometric_backend import GeometricIKSolver

        return GeometricIKSolver(config)

    raise ValueError(f"Unknown IK backend: {backend}")
