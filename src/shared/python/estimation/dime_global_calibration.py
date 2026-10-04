"""DIME-08: Observable Global Calibration and Consistent Prior Updates.

Provides outer-loop global calibration over observable sensor, kinematic, and
dynamic parameters while enforcing physical gauges, identifiability analysis,
inertia realizability, and frozen-prior revision guards (#11421, #11429).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.shared.python.core.contracts import PreconditionError, require
from src.shared.python.estimation.identifiability import (
    IdentifiabilityReport,
    ParameterSpec,
    probe_identifiability,
)


class PhysicalGauge(str, Enum):
    """Physical reference gauges required to anchor unobservable freedoms."""

    METRIC_SCALE = "metric_scale"
    GRAVITY_FRAME = "gravity_frame"
    MASS_ANCHOR = "mass_anchor"
    MEASURED_FORCE_ANCHOR = "measured_force_anchor"


class CalibrationParameterKind(str, Enum):
    """Subsystem classification for global calibration parameters."""

    CAMERA_EXTRINSICS = "camera_extrinsics"
    CAMERA_INTRINSICS = "camera_intrinsics"
    TIME_OFFSET = "time_offset"
    MARKER_OFFSET = "marker_offset"
    BODY_GEOMETRY = "body_geometry"
    BODY_INERTIA = "body_inertia"
    CONTACT_PARAMETER = "contact_parameter"


class RankDeficiencyPolicy(str, Enum):
    """Handling strategy for unobservable parameter combinations."""

    FAIL_CLOSED = "fail_closed"
    FREEZE_NULLSPACE = "freeze_nullspace"
    PROFILE_BOUNDS = "profile_bounds"


@dataclass(frozen=True)
class CalibrationParameter:
    """One bounded physical parameter candidate for global calibration."""

    name: str
    kind: CalibrationParameterKind
    value: float
    nominal_value: float
    bounds: tuple[float, float]
    sigma: float
    is_anchored: bool = False

    def __post_init__(self) -> None:
        require(bool(self.name.strip()), "Parameter name must be non-empty")
        require(np.isfinite(self.value), f"value must be finite, got {self.value}")
        require(
            np.isfinite(self.nominal_value),
            f"nominal_value must be finite, got {self.nominal_value}",
        )
        require(
            np.isfinite(self.bounds[0]),
            f"bounds.lower must be finite, got {self.bounds[0]}",
        )
        require(
            np.isfinite(self.bounds[1]),
            f"bounds.upper must be finite, got {self.bounds[1]}",
        )
        require(
            self.bounds[0] <= self.bounds[1],
            f"Lower bound {self.bounds[0]} exceeds upper bound {self.bounds[1]}",
        )
        require(self.sigma > 0.0, f"sigma must be strictly positive, got {self.sigma}")
        require(
            self.bounds[0] <= self.value <= self.bounds[1],
            f"Value {self.value} outside bounds {self.bounds}",
        )


@dataclass(frozen=True)
class PhysicalGaugePolicy:
    """Declared physical reference anchors present in the observation setup."""

    active_gauges: frozenset[PhysicalGauge] = field(default_factory=frozenset)

    def __init__(
        self,
        active_gauges: Sequence[PhysicalGauge]
        | set[PhysicalGauge]
        | frozenset[PhysicalGauge] = (),
    ) -> None:
        object.__setattr__(self, "active_gauges", frozenset(active_gauges))

    @property
    def has_metric_scale(self) -> bool:
        return PhysicalGauge.METRIC_SCALE in self.active_gauges

    @property
    def has_mass_or_force(self) -> bool:
        return (
            PhysicalGauge.MASS_ANCHOR in self.active_gauges
            or PhysicalGauge.MEASURED_FORCE_ANCHOR in self.active_gauges
        )


def validate_physical_inertia(tensor: np.ndarray) -> np.ndarray:
    """Verify that a 3x3 inertia matrix is physically realizable.

    Enforces:
    1. Square 3x3 symmetric matrix.
    2. Strict positive definiteness (all eigenvalues > 0).
    3. Classical triangle inequalities:
       I_xx + I_yy >= I_zz,  I_yy + I_zz >= I_xx,  I_zz + I_xx >= I_yy.
    """
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise PreconditionError("Inertia tensor must be finite 3x3 matrix")

    if not np.allclose(arr, arr.T, atol=1e-7):
        raise PreconditionError("Inertia tensor must be symmetric")

    eigvals = np.linalg.eigvalsh(arr)
    if np.any(eigvals <= 1e-9):
        raise PreconditionError("Inertia tensor is not positive definite")

    # Triangle inequalities on principal moments
    ixx, iyy, izz = arr[0, 0], arr[1, 1], arr[2, 2]
    tol = 1e-9
    if (ixx + iyy < izz - tol) or (iyy + izz < ixx - tol) or (izz + ixx < iyy - tol):
        raise PreconditionError(
            f"Inertia tensor violates triangle inequality: ixx={ixx}, iyy={iyy}, izz={izz}"
        )
    return arr


@dataclass(frozen=True)
class GlobalCalibrationProblem:
    """Configuration for outer-loop calibration and prior updates."""

    parameters: tuple[CalibrationParameter, ...]
    gauge_policy: PhysicalGaugePolicy
    residual_fn: Callable[[np.ndarray], np.ndarray] | None = None
    is_monocular: bool = False
    has_measured_contact_force: bool = False
    is_prior_frozen: bool = False
    prior_revision_tagged: bool = False
    inner_loop_latency_s: float = 0.0
    rank_deficiency_policy: RankDeficiencyPolicy = RankDeficiencyPolicy.FREEZE_NULLSPACE
    convergence_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        require(len(self.parameters) > 0, "At least one parameter required")
        names = [p.name for p in self.parameters]
        require(len(names) == len(set(names)), "Parameter names must be unique")


@dataclass(frozen=True)
class GlobalCalibrationResult:
    """Structured receipt from outer-loop global calibration."""

    success: bool
    parameter_values: dict[str, float]
    locked_parameters: tuple[str, ...]
    cost_breakdown: dict[str, float]
    inner_loop_latency_s: float
    outer_loop_time_s: float
    total_time_s: float
    parameter_revision: str
    identifiability_report: IdentifiabilityReport | None = None

    @property
    def total_cost(self) -> float:
        """Return total optimization cost."""
        return float(self.cost_breakdown.get("total_cost", 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "global-calibration-result-v1",
            "success": self.success,
            "parameter_values": dict(self.parameter_values),
            "locked_parameters": list(self.locked_parameters),
            "cost_breakdown": dict(self.cost_breakdown),
            "inner_loop_latency_s": float(self.inner_loop_latency_s),
            "outer_loop_time_s": float(self.outer_loop_time_s),
            "total_time_s": float(self.total_time_s),
            "parameter_revision": str(self.parameter_revision),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GlobalCalibrationResult:
        require(
            data.get("schema_version") == "global-calibration-result-v1",
            "Invalid schema version",
        )
        return cls(
            success=bool(data["success"]),
            parameter_values={
                str(k): float(v) for k, v in data["parameter_values"].items()
            },
            locked_parameters=tuple(str(k) for k in data["locked_parameters"]),
            cost_breakdown={
                str(k): float(v) for k, v in data["cost_breakdown"].items()
            },
            inner_loop_latency_s=float(data["inner_loop_latency_s"]),
            outer_loop_time_s=float(data["outer_loop_time_s"]),
            total_time_s=float(data["total_time_s"]),
            parameter_revision=str(data["parameter_revision"]),
            identifiability_report=None,
        )


def calibrate_global_parameters(
    problem: GlobalCalibrationProblem,
) -> GlobalCalibrationResult:
    """Execute outer-loop parameter calibration with identifiability checks.

    Fail-closed checks:
    1. Monocular scale ambiguity: monocular cameras cannot determine metric
       body dimensions without a metric scale anchor.
    2. Mass/torque scaling ambiguity: motion alone cannot determine body mass
       and actuator torque independently without mass or force anchor.
    3. Frozen prior protection: prior cannot be silently modified without tag.
    4. Rank-deficient parameter combinations: detected via SVD and locked.
    """
    t_start = time.perf_counter()

    # 1. Monocular scale ambiguity check
    has_body_geom = any(
        p.kind == CalibrationParameterKind.BODY_GEOMETRY for p in problem.parameters
    )
    if (
        problem.is_monocular
        and has_body_geom
        and not problem.gauge_policy.has_metric_scale
    ):
        raise PreconditionError(
            "Unanchored monocular observation cannot determine absolute metric scale; "
            "monocular scale ambiguity requires a metric scale anchor"
        )

    # 2. Mass/torque scaling check
    has_mass = any(
        p.kind == CalibrationParameterKind.BODY_INERTIA for p in problem.parameters
    )
    has_torque = any(
        p.kind
        in (
            CalibrationParameterKind.CONTACT_PARAMETER,
            CalibrationParameterKind.BODY_GEOMETRY,
        )
        for p in problem.parameters
    )
    if (
        has_mass
        and has_torque
        and not problem.gauge_policy.has_mass_or_force
        and not problem.has_measured_contact_force
    ):
        raise PreconditionError(
            "Cannot claim unique forces without mass or force anchor; "
            "mass/torque scaling ambiguity requires mass or measured-force anchor"
        )

    # 3. Frozen prior protection check
    if problem.is_prior_frozen and not problem.prior_revision_tagged:
        raise PreconditionError(
            "Cannot apply calibration changes beneath a frozen prior without explicit revision"
        )

    # Prepare parameter vectors
    names = [p.name for p in problem.parameters]
    nominal_vals = np.array(
        [p.nominal_value for p in problem.parameters], dtype=np.float64
    )
    x0 = np.array([p.value for p in problem.parameters], dtype=np.float64)
    lower_bounds = np.array([p.bounds[0] for p in problem.parameters], dtype=np.float64)
    upper_bounds = np.array([p.bounds[1] for p in problem.parameters], dtype=np.float64)
    sigmas = np.array([p.sigma for p in problem.parameters], dtype=np.float64)

    # If no observation residual is provided, just return initial values
    if problem.residual_fn is None:
        t_outer = time.perf_counter() - t_start
        return GlobalCalibrationResult(
            success=True,
            parameter_values={p.name: p.value for p in problem.parameters},
            locked_parameters=(),
            cost_breakdown={
                "data_cost": 0.0,
                "prior_cost": 0.0,
                "total_cost": 0.0,
            },
            inner_loop_latency_s=problem.inner_loop_latency_s,
            outer_loop_time_s=t_outer,
            total_time_s=problem.inner_loop_latency_s + t_outer,
            parameter_revision="rev-001",
        )

    # 4. SVD Identifiability probe
    res0 = problem.residual_fn(x0)
    spec = ParameterSpec(names=tuple(names))
    report = probe_identifiability(problem.residual_fn, x0, spec)

    locked: list[str] = []
    free_indices: list[int] = []

    # Check for rank deficiency
    if report.rank < len(names):
        if problem.rank_deficiency_policy == RankDeficiencyPolicy.FAIL_CLOSED:
            raise PreconditionError(
                f"Parameter block is rank-deficient (rank {report.rank} < {len(names)}) "
                "with unobservable null space"
            )
        # Lock unobservable parameters: inspect right singular vectors for null directions
        null_mask = np.zeros(len(names), dtype=bool)
        if report.nullspace_directions:
            for direction in report.nullspace_directions.values():
                null_vec = np.abs(direction)
                max_idx = int(np.argmax(null_vec))
                null_mask[max_idx] = True
        else:
            for i in range(report.rank, len(names)):
                null_vec = np.abs(report.right_singular_vectors[:, i])
                max_idx = int(np.argmax(null_vec))
                null_mask[max_idx] = True

        for i, name in enumerate(names):
            if null_mask[i]:
                locked.append(name)
            else:
                free_indices.append(i)
    else:
        free_indices = list(range(len(names)))

    # If all parameters are locked, return nominal
    if not free_indices:
        t_outer = time.perf_counter() - t_start
        return GlobalCalibrationResult(
            success=True,
            parameter_values={p.name: p.nominal_value for p in problem.parameters},
            locked_parameters=tuple(locked),
            cost_breakdown={
                "data_cost": float(0.5 * np.sum(res0**2)),
                "prior_cost": 0.0,
                "total_cost": float(0.5 * np.sum(res0**2)),
            },
            inner_loop_latency_s=problem.inner_loop_latency_s,
            outer_loop_time_s=t_outer,
            total_time_s=problem.inner_loop_latency_s + t_outer,
            parameter_revision="rev-001",
            identifiability_report=report,
        )

    # Optimize free parameters
    x_free_0 = x0[free_indices]
    lb_free = lower_bounds[free_indices]
    ub_free = upper_bounds[free_indices]

    def joint_residual(x_free: np.ndarray) -> np.ndarray:
        x_full = np.copy(nominal_vals)
        x_full[free_indices] = x_free
        return np.asarray(problem.residual_fn(x_full), dtype=np.float64)

    opt = least_squares(
        joint_residual,
        x_free_0,
        bounds=(lb_free, ub_free),
        ftol=problem.convergence_tolerance,
        xtol=problem.convergence_tolerance,
        gtol=problem.convergence_tolerance,
    )

    x_final = np.copy(nominal_vals)
    x_final[free_indices] = opt.x

    # Compute final costs
    data_res_final = np.asarray(problem.residual_fn(x_final), dtype=np.float64)
    data_cost = float(0.5 * np.sum(data_res_final**2))
    prior_cost = 0.0
    total_cost = data_cost + prior_cost

    t_outer = time.perf_counter() - t_start

    return GlobalCalibrationResult(
        success=bool(opt.success),
        parameter_values={names[i]: float(x_final[i]) for i in range(len(names))},
        locked_parameters=tuple(locked),
        cost_breakdown={
            "data_cost": data_cost,
            "prior_cost": prior_cost,
            "total_cost": total_cost,
        },
        inner_loop_latency_s=problem.inner_loop_latency_s,
        outer_loop_time_s=t_outer,
        total_time_s=problem.inner_loop_latency_s + t_outer,
        parameter_revision="rev-001",
        identifiability_report=report,
    )
