"""
Base Protocol and factory for Motion Matching solvers.

Part of issue #4568. Defines the unified motion matching interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Protocol

import numpy as np

from ..contracts import (
    JointTrajectory,
    MotionMatchingResult as ContractMotionMatchingResult,
    MotionTrajectory,
    MuscleActivationTrajectory,
    SkeletonRig,
    TorqueTrajectory,
)


def _require_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_finite_nonnegative(value: float, field_name: str) -> None:
    number = _coerce_float(value, field_name)
    if number < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _require_finite_positive(value: float, field_name: str) -> None:
    number = _coerce_float(value, field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _coerce_float(value: float, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _require_mapping(value: object, field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be mapping-like")


def _trajectory_id(trajectory: JointTrajectory) -> str:
    return str(getattr(trajectory, "id", "<unknown>"))


def _require_non_empty_joint_trajectory(
    trajectory: JointTrajectory, field_name: str
) -> None:
    frames = getattr(trajectory, "frames", None)
    if not frames:
        raise ValueError(f"{field_name} must contain at least one frame")


def _validate_monotonic_finite_times(
    frames: list[Any],
    field_name: str,
) -> None:
    previous: float | None = None
    for frame_index, frame in enumerate(frames):
        timestamp = _coerce_float(
            frame.timestamp, f"{field_name} frame {frame_index} timestamp"
        )
        if previous is not None and timestamp <= previous:
            raise ValueError(
                f"{field_name} timestamps must be strictly monotonic; "
                f"frame {frame_index - 1}={previous}, frame {frame_index}={timestamp}"
            )
        previous = timestamp


def _validate_trajectory_shape_and_values(
    reference: JointTrajectory,
    tracked: JointTrajectory,
    *,
    field_name: str,
) -> None:
    _require_non_empty_joint_trajectory(reference, "reference")
    _require_non_empty_joint_trajectory(tracked, field_name)
    _validate_monotonic_finite_times(reference.frames, "reference")

    expected_frames = len(reference.frames)
    actual_frames = len(tracked.frames)
    if actual_frames != expected_frames:
        raise ValueError(
            f"{field_name} shape mismatch for reference '{_trajectory_id(reference)}' "
            f"and tracked '{_trajectory_id(tracked)}': expected {expected_frames} "
            f"frames, actual {actual_frames}"
        )

    _validate_monotonic_finite_times(tracked.frames, field_name)

    for frame_index, (ref_frame, tracked_frame) in enumerate(
        zip(reference.frames, tracked.frames, strict=True)
    ):
        expected_dofs = len(ref_frame.q)
        actual_dofs = len(tracked_frame.q)
        if expected_dofs <= 0:
            raise ValueError(
                f"reference frame {frame_index} must contain at least one q DOF"
            )
        if actual_dofs != expected_dofs:
            raise ValueError(
                f"{field_name} shape mismatch at frame {frame_index} for "
                f"reference '{_trajectory_id(reference)}' and tracked "
                f"'{_trajectory_id(tracked)}': expected {expected_dofs} DOFs, "
                f"actual {actual_dofs}"
            )
        if not all(np.isfinite(value) for value in ref_frame.q):
            raise ValueError(
                f"reference frame {frame_index} q contains non-finite values"
            )
        if not all(np.isfinite(value) for value in tracked_frame.q):
            raise ValueError(
                f"{field_name} frame {frame_index} q contains non-finite values"
            )


def _validate_matching_time_grid(
    reference: JointTrajectory,
    trajectory: JointTrajectory,
    *,
    field_name: str,
) -> None:
    for frame_index, (ref_frame, candidate_frame) in enumerate(
        zip(reference.frames, trajectory.frames, strict=True)
    ):
        reference_time = float(ref_frame.timestamp)
        candidate_time = float(candidate_frame.timestamp)
        if not np.isclose(reference_time, candidate_time, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{field_name} time grid mismatch at frame {frame_index}: "
                f"expected {reference_time}, actual {candidate_time}"
            )


def _validate_aligned_signal_trajectory(
    reference: JointTrajectory,
    payload: TorqueTrajectory | MuscleActivationTrajectory,
    *,
    field_name: str,
) -> None:
    frames = list(getattr(payload, "frames", []))
    if not frames:
        raise ValueError(f"{field_name} must contain at least one frame")
    _validate_monotonic_finite_times(frames, field_name)
    expected_frames = len(reference.frames)
    actual_frames = len(frames)
    if actual_frames != expected_frames:
        raise ValueError(
            f"{field_name} frame count mismatch: expected {expected_frames} frames, "
            f"actual {actual_frames}"
        )

    expected_width: int | None = None
    values_name = "values"
    if isinstance(payload, TorqueTrajectory):
        expected_width = len(payload.rig_joint_names)
        values_name = "tau"
        if expected_width != reference.skeleton.num_dofs:
            raise ValueError(
                f"{field_name} rig_joint_names length {expected_width} does not match "
                f"reference DOFs {reference.skeleton.num_dofs}"
            )
    elif isinstance(payload, MuscleActivationTrajectory):
        expected_width = len(payload.muscle_names)
        values_name = "activations"

    for frame_index, (ref_frame, payload_frame) in enumerate(
        zip(reference.frames, frames, strict=True)
    ):
        payload_time = float(payload_frame.timestamp)
        reference_time = float(ref_frame.timestamp)
        if not np.isclose(reference_time, payload_time, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"{field_name} time grid mismatch at frame {frame_index}: "
                f"expected {reference_time}, actual {payload_time}"
            )
        values = list(getattr(payload_frame, values_name))
        if not values:
            raise ValueError(
                f"{field_name} frame {frame_index} {values_name} must not be empty"
            )
        if expected_width is not None and len(values) != expected_width:
            raise ValueError(
                f"{field_name} frame {frame_index} {values_name} length "
                f"{len(values)} != expected {expected_width}"
            )
        if not all(np.isfinite(value) for value in values):
            raise ValueError(
                f"{field_name} frame {frame_index} {values_name} contains "
                "non-finite values"
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

    def __post_init__(self) -> None:
        for weight_field in fields(self):
            _require_finite_nonnegative(
                getattr(self, weight_field.name), weight_field.name
            )


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

    def __post_init__(self) -> None:
        _require_non_empty_string(self.id, "id")
        _require_non_empty_joint_trajectory(self.reference, "reference.frames")
        if self.time_horizon is None:
            self.time_horizon = self.reference.duration
        _require_finite_positive(self.time_horizon, "time_horizon")
        _require_finite_positive(self.integrator_step, "integrator_step")
        _require_positive_int(self.max_iterations, "max_iterations")
        if self.reference.skeleton.num_dofs != self.rig.num_dofs:
            raise ValueError(
                "rig must be compatible with reference: "
                f"reference DOFs={self.reference.skeleton.num_dofs}, "
                f"rig DOFs={self.rig.num_dofs}"
            )


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
    torque_trajectory: TorqueTrajectory | None = None
    activation_trajectory: MuscleActivationTrajectory | None = None
    residual_report: dict | None = None
    fit_metrics: dict | None = None
    solve_time: float | None = None
    message: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, "request_id")
        if self.residual_report is not None:
            _require_mapping(self.residual_report, "residual_report")
        if self.fit_metrics is not None:
            _require_mapping(self.fit_metrics, "fit_metrics")
        _require_mapping(self.metadata, "metadata")

        has_payload = (
            self.tracked_trajectory is not None
            or self.torque_trajectory is not None
            or self.activation_trajectory is not None
        )
        if self.success and not has_payload:
            raise ValueError(
                "success=True requires at least one matched payload: "
                "tracked_trajectory, torque_trajectory, or activation_trajectory"
            )
        if not self.success and not (self.message and self.message.strip()):
            raise ValueError("message must explain failed motion matching results")

    def to_contract(self) -> ContractMotionMatchingResult:
        """Convert to contract MotionMatchingResult.

        Constructs a ``MotionTrajectory`` wrapper from the internal ``tracked_trajectory``.
        """
        matched_trajectory = None
        if self.tracked_trajectory is not None:
            matched_trajectory = MotionTrajectory(
                id=self.tracked_trajectory.id,
                skeleton=self.tracked_trajectory.skeleton,
                trajectory=self.tracked_trajectory,
            )
        return ContractMotionMatchingResult(
            request_id=self.request_id,
            success=self.success,
            matched_trajectory=matched_trajectory,
            torques=self.torque_trajectory,
            activations=self.activation_trajectory,
            error_metrics=self.fit_metrics or {},
            message=self.message,
            metadata=self.metadata,
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
        _validate_trajectory_shape_and_values(
            reference, tracked, field_name="metric trajectory"
        )
        for ref_frame, track_frame in zip(
            reference.frames, tracked.frames, strict=True
        ):
            for ref_q, track_q in zip(ref_frame.q, track_frame.q, strict=True):
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
        _validate_trajectory_shape_and_values(
            reference, tracked, field_name="metric trajectory"
        )
        for ref_frame, track_frame in zip(
            reference.frames, tracked.frames, strict=True
        ):
            for ref_q, track_q in zip(ref_frame.q, track_frame.q, strict=True):
                residuals.append(abs(ref_q - track_q))

        return {
            "mean_residual": float(np.mean(residuals)),
            "max_residual": float(np.max(residuals)),
            "std_residual": float(np.std(residuals)),
            "num_frames": len(reference.frames),
        }

    def _validate_result(
        self,
        reference: JointTrajectory,
        result: MotionMatchingResult,
    ) -> bool:
        """
        Validate motion matching result satisfies DbC postconditions.

        Postconditions:
        - Torque/activation finite
        - Time grid matches reference

        Raises:
            ValueError: If the first broken invariant is found.
        """
        _require_non_empty_joint_trajectory(reference, "reference")
        _validate_monotonic_finite_times(reference.frames, "reference")
        if result.tracked_trajectory is not None:
            _validate_trajectory_shape_and_values(
                reference,
                result.tracked_trajectory,
                field_name="tracked_trajectory",
            )
            _validate_matching_time_grid(
                reference,
                result.tracked_trajectory,
                field_name="tracked_trajectory",
            )
        if result.torque_trajectory is not None:
            _validate_aligned_signal_trajectory(
                reference,
                result.torque_trajectory,
                field_name="torque_trajectory",
            )
        if result.activation_trajectory is not None:
            _validate_aligned_signal_trajectory(
                reference,
                result.activation_trajectory,
                field_name="activation_trajectory",
            )

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
