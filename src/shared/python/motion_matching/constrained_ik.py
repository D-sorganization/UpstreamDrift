"""Shared constrained inverse kinematics protocols and data contracts (Packet P2, #10277).

Defines engine-neutral trajectory requests, options, rate audits, and results
with strict DbC preconditions, immutability, and physical-time invariants.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.tour_capture_contract import TourCapture

if TYPE_CHECKING:
    from src.engines.physics_engines.pinocchio.python.pink_tasks import (
        FrameResiduals,
        StanceClosurePolicy,
    )

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class IKTrajectoryRequest:
    """Immutable request specifying capture targets, timing, and model identity."""

    initial_q: Array
    time_s: Array
    marker_targets: Array
    validity_mask: BoolArray
    labels: tuple[str, ...]
    model_name: str
    calibration_identity: str | None = None
    posture_target: Mapping[str, float] | Array | None = None
    policy: StanceClosurePolicy | None = None
    cancellation_token: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        init_q = np.asarray(self.initial_q, dtype=np.float64).copy()
        time = np.asarray(self.time_s, dtype=np.float64).copy()
        targets = np.asarray(self.marker_targets, dtype=np.float64).copy()
        valid = np.asarray(self.validity_mask, dtype=bool).copy()
        labels_tuple = tuple(self.labels)

        self._validate_arrays(init_q, time, targets, valid, labels_tuple)

        init_q.setflags(write=False)
        time.setflags(write=False)
        targets.setflags(write=False)
        valid.setflags(write=False)

        object.__setattr__(self, "initial_q", init_q)
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "marker_targets", targets)
        object.__setattr__(self, "validity_mask", valid)
        object.__setattr__(self, "labels", labels_tuple)

    @staticmethod
    def _validate_arrays(
        init_q: Array,
        time: Array,
        targets: Array,
        valid: BoolArray,
        labels: tuple[str, ...],
    ) -> None:
        """Validate shapes, monotonicity, uniqueness, and finite bounds."""
        if init_q.ndim != 1 or not np.isfinite(init_q).all():
            raise ValueError("initial_q must be a finite 1D array")
        if time.ndim != 1 or time.size == 0 or time[0] != 0.0:
            raise ValueError("Capture time must start at zero and increase strictly")
        if not np.all(np.isfinite(time)):
            raise ValueError("Capture time must start at zero and increase strictly")
        dt = np.diff(time)
        if dt.size > 0 and not np.all(dt > 0.0):
            raise ValueError("Capture time must start at zero and increase strictly")
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("labels must be unique and non-empty")
        if targets.ndim != 3 or targets.shape[0] != time.size:
            raise ValueError(
                f"marker_targets frames {targets.shape[0] if targets.ndim >= 1 else 0} "
                f"must match time frames {time.size}"
            )
        if targets.shape[1] != len(labels) or targets.shape[2] != 3:
            raise ValueError(
                f"marker_targets markers ({targets.shape[1]}, {targets.shape[2]}) "
                f"must match (len(labels), 3)"
            )
        if valid.shape != (time.size, len(labels)):
            raise ValueError(
                f"validity_mask shape {valid.shape} must match (frames, markers)"
            )
        if not np.isfinite(targets[valid]).all():
            raise ValueError("Every valid marker target point must be finite")

    @property
    def num_frames(self) -> int:
        return int(self.time_s.size)

    @property
    def num_markers(self) -> int:
        return len(self.labels)

    @classmethod
    def from_capture(
        cls,
        capture: TourCapture,
        initial_q: Array,
        model_name: str,
        calibration_identity: str | None = None,
        posture_target: Mapping[str, float] | Array | None = None,
        policy: StanceClosurePolicy | None = None,
        cancellation_token: Callable[[], bool] | None = None,
    ) -> IKTrajectoryRequest:
        """Construct an IKTrajectoryRequest directly from a TourCapture instance."""
        return cls(
            initial_q=initial_q,
            time_s=capture.time_s,
            marker_targets=capture.points_m,
            validity_mask=capture.valid,
            labels=capture.labels,
            model_name=model_name,
            calibration_identity=calibration_identity or capture.source_sha256,
            posture_target=posture_target,
            policy=policy,
            cancellation_token=cancellation_token,
        )


@dataclass(frozen=True)
class IKOptions:
    """Options controlling numerical vs physical solver execution."""

    step_mode: str = "physical"
    solver: str = "quadprog"
    max_iterations: int = 10
    damping: float = 1e-6
    tolerance: float = 1e-4
    limit_policy: str = "enforce"
    dt_numerical: float = 1.0 / 360.0
    weld_translation_tolerance_m: float = 0.005
    weld_rotation_tolerance_rad: float = 0.05
    marker_tolerance_m: float = 0.03

    def __post_init__(self) -> None:
        if self.step_mode not in ("physical", "projection"):
            raise ValueError(
                f"step_mode must be 'physical' or 'projection', got {self.step_mode}"
            )
        if not self.solver or not isinstance(self.solver, str):
            raise ValueError("solver must be a non-empty string")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.damping < 0.0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")
        if self.tolerance <= 0.0:
            raise ValueError(f"tolerance must be > 0, got {self.tolerance}")
        if self.limit_policy not in ("enforce", "ignore"):
            raise ValueError(
                f"limit_policy must be 'enforce' or 'ignore', got {self.limit_policy}"
            )
        if self.dt_numerical <= 0.0:
            raise ValueError(f"dt_numerical must be > 0, got {self.dt_numerical}")
        if self.weld_translation_tolerance_m <= 0.0:
            raise ValueError(
                f"weld_translation_tolerance_m must be > 0, got {self.weld_translation_tolerance_m}"
            )
        if self.weld_rotation_tolerance_rad <= 0.0:
            raise ValueError(
                f"weld_rotation_tolerance_rad must be > 0, got {self.weld_rotation_tolerance_rad}"
            )
        if self.marker_tolerance_m <= 0.0:
            raise ValueError(
                f"marker_tolerance_m must be > 0, got {self.marker_tolerance_m}"
            )


@dataclass(frozen=True)
class FrameRateAudit:
    """Audited kinematic velocity and joint limit comparison for one frame."""

    frame_index: int
    dt_s: float
    joint_velocities: Array
    max_velocity_ratio: float
    exceeded_joints: tuple[str, ...]

    def __post_init__(self) -> None:
        v = np.asarray(self.joint_velocities, dtype=np.float64)
        v.setflags(write=False)
        object.__setattr__(self, "joint_velocities", v)

    @classmethod
    def compute(
        cls,
        frame_index: int,
        dt_s: float,
        delta_q: Array,
        velocity_limits: Array | None,
        joint_names: Sequence[str] | None,
    ) -> FrameRateAudit:
        """Compute joint velocities and check against declared limits."""
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")
        velocities = np.asarray(delta_q, dtype=np.float64) / dt_s
        exceeded: list[str] = []
        max_ratio = 0.0

        if velocity_limits is not None:
            limits = np.asarray(velocity_limits, dtype=np.float64)
            for i in range(velocities.size):
                lim = abs(float(limits[i])) if i < limits.size else float("inf")
                if lim > 0:
                    ratio = abs(float(velocities[i])) / lim
                    if ratio > max_ratio:
                        max_ratio = ratio
                    if ratio > 1.0:
                        name = (
                            joint_names[i]
                            if joint_names and i < len(joint_names)
                            else f"joint_{i}"
                        )
                        exceeded.append(name)

        return cls(
            frame_index=frame_index,
            dt_s=float(dt_s),
            joint_velocities=velocities,
            max_velocity_ratio=float(max_ratio),
            exceeded_joints=tuple(exceeded),
        )


@dataclass(frozen=True)
class IKTrajectoryResult:
    """Full trajectory result containing configurations, diagnostics, and audits."""

    configurations: Array
    frame_success: BoolArray
    frame_residuals: tuple[FrameResiduals, ...]
    rate_audits: tuple[FrameRateAudit, ...]
    timing_ms: Array
    total_time_ms: float
    backend_name: str
    passed: bool
    first_failed_frame: int | None = None
    cancelled: bool = False
    failure_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        q = np.asarray(self.configurations, dtype=np.float64).copy()
        success = np.asarray(self.frame_success, dtype=bool).copy()
        timing = np.asarray(self.timing_ms, dtype=np.float64).copy()

        if self.passed and not np.all(success):
            raise ValueError("passed cannot be True if any frame failed")
        if self.passed and self.cancelled:
            raise ValueError("passed cannot be True if run was cancelled")

        q.setflags(write=False)
        success.setflags(write=False)
        timing.setflags(write=False)

        object.__setattr__(self, "configurations", q)
        object.__setattr__(self, "frame_success", success)
        object.__setattr__(self, "timing_ms", timing)


@runtime_checkable
class ConstrainedIKBackend(Protocol):
    """Protocol for constrained multi-frame inverse kinematics solvers."""

    @property
    def backend_name(self) -> str:
        """Declared identifier of the solver backend."""
        ...

    def solve_trajectory(
        self, request: IKTrajectoryRequest, options: IKOptions
    ) -> IKTrajectoryResult:
        """Solve constrained IK across all frames in request."""
        ...

    def audit_trajectory(
        self, trajectory: Array, request: IKTrajectoryRequest
    ) -> IKTrajectoryResult:
        """Audit an existing or smoothed trajectory against constraints and limits."""
        ...
