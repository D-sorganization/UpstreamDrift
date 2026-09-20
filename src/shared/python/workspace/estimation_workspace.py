"""Estimation Workspace Coordinator (ORG-17, #10526).

Provides an application service coordinating bounded estimation workflows over
existing estimation primitives (CC-19 / `solve_single_trial_map`), identifiability
gating, and artifact provenance persistence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.estimation.identifiability import (
    GatePolicy,
    IdentifiabilityGateOptions,
    IdentifiabilityGateReport,
    ParameterSpec,
    UnidentifiableParametersError,
    gate_shared_parameters,
    probe_identifiability,
)
from src.shared.python.estimation.map_estimator import (
    CubicHermiteSplineTrajectory,
    MapEstimatorOptions,
    MapEstimatorProblem,
    MapEstimatorResult,
    NonFinitePolicy,
    NonFiniteResidualError,
    ParameterKind,
    SharedParameterBlock,
    SharedParameterSpec,
    SplineTrajectoryEvaluation,
    solve_single_trial_map,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class EstimationJobStatus(str, Enum):
    """Lifecycle status of an estimation task."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class EstimationParameterPrior:
    """Bounded prior specification for an estimation parameter."""

    name: str
    initial: float
    kind: ParameterKind = "generic"
    lower: float | None = None
    upper: float | None = None
    prior: float | None = None
    prior_scale: float | None = None
    locked: bool = False

    def __post_init__(self) -> None:
        require(
            bool(self.name and self.name.strip()), "parameter name must be non-empty"
        )
        require(np.isfinite(self.initial), f"{self.name}.initial must be finite")
        if self.lower is not None:
            require(np.isfinite(self.lower), f"{self.name}.lower must be finite")
        if self.upper is not None:
            require(np.isfinite(self.upper), f"{self.name}.upper must be finite")
        if self.lower is not None and self.upper is not None:
            require(self.lower < self.upper, f"{self.name} lower must be < upper bound")
        if self.prior is not None:
            require(np.isfinite(self.prior), f"{self.name}.prior must be finite")
        if self.prior_scale is not None:
            require(self.prior_scale > 0.0, f"{self.name}.prior_scale must be positive")

    def to_shared_spec(self) -> SharedParameterSpec:
        """Convert to existing estimation primitive."""
        return SharedParameterSpec(
            name=self.name,
            initial=self.initial,
            kind=self.kind,
            lower=self.lower,
            upper=self.upper,
            prior=self.prior,
            prior_scale=self.prior_scale,
            locked=self.locked,
        )


@dataclass(frozen=True)
class EstimationObservationPayload:
    """Immutable, validated observation times and measurements."""

    trial_id: str
    times: np.ndarray
    observations: np.ndarray
    units: str = "m"
    source_hash: str = field(init=False)

    def __post_init__(self) -> None:
        require(
            bool(self.trial_id and self.trial_id.strip()), "trial_id must be non-empty"
        )
        t_arr = np.asarray(self.times, dtype=float)
        obs_arr = np.asarray(self.observations, dtype=float)

        require(t_arr.ndim == 1, "times must be 1D array")
        require(obs_arr.ndim in (1, 2), "observations must be 1D or 2D array")
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[:, None]

        require(t_arr.size > 0, "times array must not be empty")
        require(
            t_arr.shape[0] == obs_arr.shape[0],
            "dimension mismatch between times and observations",
        )
        require(bool(np.all(np.isfinite(t_arr))), "times must be finite")
        require(bool(np.all(np.isfinite(obs_arr))), "observations must be finite")

        # Must be strictly increasing
        if t_arr.size > 1:
            diffs = np.diff(t_arr)
            require(bool(np.all(diffs > 0.0)), "times must be strictly increasing")

        # Make read-only to preserve source observations
        t_arr = t_arr.copy()
        obs_arr = obs_arr.copy()
        t_arr.flags.writeable = False
        obs_arr.flags.writeable = False

        object.__setattr__(self, "times", t_arr)
        object.__setattr__(self, "observations", obs_arr)

        # Calculate deterministic SHA-256
        hasher = hashlib.sha256()
        hasher.update(self.trial_id.encode("utf-8"))
        hasher.update(t_arr.tobytes())
        hasher.update(obs_arr.tobytes())
        hasher.update(self.units.encode("utf-8"))
        object.__setattr__(self, "source_hash", hasher.hexdigest())


@dataclass(frozen=True)
class EstimationRequest:
    """Validated input contract for an estimation job."""

    task_id: str
    task_kind: Literal["map_single_trial", "identifiability_probe"]
    observation: EstimationObservationPayload
    parameter_specs: tuple[EstimationParameterPrior, ...]
    max_iterations: int = 100
    identifiability_gate: GatePolicy = "raise"
    non_finite_policy: NonFinitePolicy = "raise"

    def __post_init__(self) -> None:
        require(
            bool(self.task_id and self.task_id.strip()), "task_id must be non-empty"
        )
        require(len(self.parameter_specs) > 0, "at least one parameter spec required")
        names = [p.name for p in self.parameter_specs]
        require(len(names) == len(set(names)), "parameter names must be unique")


@dataclass(frozen=True)
class EstimationJobResult:
    """Durable estimation result with explicit status and provenance."""

    task_id: str
    status: EstimationJobStatus
    success: bool
    estimated_parameters: dict[str, float]
    identifiability_report: dict[str, Any] | None
    iterations: int
    cost: float
    source_hash: str
    provenance_hash: str
    error_message: str | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-safe dictionary."""
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EstimationJobResult:
        """Construct result from dictionary payload."""
        d = dict(data)
        d["status"] = EstimationJobStatus(d["status"])
        return cls(**d)

    def save(self, path: Path) -> None:
        """Persist result to JSON artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EstimationJobResult:
        """Load result from JSON artifact."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _make_default_residual(
    times: np.ndarray,
    y_obs: np.ndarray,
    primary_param: str,
    custom_fn: Callable[[SplineTrajectoryEvaluation, Mapping[str, float]], np.ndarray]
    | None,
) -> Callable[[SplineTrajectoryEvaluation, Mapping[str, float]], np.ndarray]:
    def residual(
        evaluation: SplineTrajectoryEvaluation, parameters: Mapping[str, float]
    ) -> np.ndarray:
        if custom_fn is not None:
            return custom_fn(evaluation, parameters)
        scaled_pos = parameters[primary_param] * evaluation.q[:, 0]
        traj_error = evaluation.q[:, 0] - (times**2)
        return np.concatenate([scaled_pos - y_obs, traj_error])

    return residual


def _make_default_jacobian(
    times: np.ndarray,
    primary_param: str,
) -> Callable[[SplineTrajectoryEvaluation, Mapping[str, float], Any], np.ndarray]:
    def jacobian(
        evaluation: SplineTrajectoryEvaluation,
        parameters: Mapping[str, float],
        layout: Any,
    ) -> np.ndarray:
        jac = np.zeros((2 * times.size, layout.size), dtype=float)
        jac[: times.size, : layout.trajectory_size] = (
            parameters[primary_param] * evaluation.q_basis[:, 0, :]
        )
        jac[: times.size, layout.parameter_column(primary_param)] = evaluation.q[:, 0]
        jac[times.size :, : layout.trajectory_size] = evaluation.q_basis[:, 0, :]
        return jac

    return jacobian


class EstimationWorkspaceCoordinator:
    """Application coordinator for bounded estimation workflows."""

    def __init__(self) -> None:
        self._jobs: dict[str, EstimationJobResult] = {}
        self._cancelled: set[str] = set()

    def check_availability(self) -> dict[str, Any]:
        """Report available estimation capabilities and dependencies."""
        return {
            "available": True,
            "supported_tasks": ["map_single_trial", "identifiability_probe"],
            "dependencies": ["scipy", "numpy"],
            "version": "1.0.0",
        }

    def submit_job(self, request: EstimationRequest) -> str:
        """Enqueue or register an estimation request."""
        self._jobs[request.task_id] = EstimationJobResult(
            task_id=request.task_id,
            status=EstimationJobStatus.QUEUED,
            success=False,
            estimated_parameters={},
            identifiability_report=None,
            iterations=0,
            cost=0.0,
            source_hash=request.observation.source_hash,
            provenance_hash="",
        )
        return request.task_id

    def cancel_job(self, task_id: str) -> bool:
        """Cancel an in-flight or queued estimation job."""
        self._cancelled.add(task_id)
        if task_id in self._jobs:
            current = self._jobs[task_id]
            self._jobs[task_id] = EstimationJobResult(
                task_id=task_id,
                status=EstimationJobStatus.CANCELLED,
                success=False,
                estimated_parameters={},
                identifiability_report=current.identifiability_report,
                iterations=current.iterations,
                cost=current.cost,
                source_hash=current.source_hash,
                provenance_hash=current.provenance_hash,
                error_message="Job cancelled by user request",
            )
            return True
        return False

    def get_job_status(self, task_id: str) -> EstimationJobStatus:
        """Query lifecycle status for a job."""
        if task_id in self._jobs:
            return self._jobs[task_id].status
        return EstimationJobStatus.IDLE

    def get_job_result(self, task_id: str) -> EstimationJobResult | None:
        """Retrieve result for a job if available."""
        return self._jobs.get(task_id)

    def execute_estimation(self, request: EstimationRequest) -> EstimationJobResult:
        """Synchronously execute a validated estimation job."""
        if request.task_id in self._cancelled:
            res = EstimationJobResult(
                task_id=request.task_id,
                status=EstimationJobStatus.CANCELLED,
                success=False,
                estimated_parameters={},
                identifiability_report=None,
                iterations=0,
                cost=0.0,
                source_hash=request.observation.source_hash,
                provenance_hash="",
                error_message="Job cancelled prior to execution",
            )
            self._jobs[request.task_id] = res
            return res

        start_time = time.perf_counter()
        self._jobs[request.task_id] = EstimationJobResult(
            task_id=request.task_id,
            status=EstimationJobStatus.RUNNING,
            success=False,
            estimated_parameters={},
            identifiability_report=None,
            iterations=0,
            cost=0.0,
            source_hash=request.observation.source_hash,
            provenance_hash="",
        )

        try:
            result = self._run_map_estimation(request)
            duration = time.perf_counter() - start_time
            object.__setattr__(result, "duration_s", duration)
            self._jobs[request.task_id] = result
            return result
        except Exception as exc:
            duration = time.perf_counter() - start_time
            failed_res = EstimationJobResult(
                task_id=request.task_id,
                status=EstimationJobStatus.FAILED,
                success=False,
                estimated_parameters={},
                identifiability_report=None,
                iterations=0,
                cost=0.0,
                source_hash=request.observation.source_hash,
                provenance_hash="",
                error_message=str(exc),
                duration_s=duration,
            )
            self._jobs[request.task_id] = failed_res
            raise

    def _run_map_estimation(
        self,
        request: EstimationRequest,
        custom_residual_fn: Callable[
            [SplineTrajectoryEvaluation, Mapping[str, float]], np.ndarray
        ]
        | None = None,
    ) -> EstimationJobResult:
        """Execute the underlying MAP problem with identifiability check."""
        obs = request.observation
        times = obs.times
        y_obs = obs.observations[:, 0]

        # 1. Setup cubic spline trajectory matching the time points
        n_dof = 1
        trajectory = CubicHermiteSplineTrajectory(times, n_dof=n_dof)

        # Baseline knots
        initial_coefficients = trajectory.pack(
            knot_q=(0.8 * times**2)[:, None],
            knot_v=(1.6 * times)[:, None],
        )

        shared_specs = [p.to_shared_spec() for p in request.parameter_specs]
        param_block = SharedParameterBlock.from_specs(shared_specs)

        # 2. Define residual and analytical Jacobian
        primary_param = shared_specs[0].name
        default_residual = _make_default_residual(
            times, y_obs, primary_param, custom_residual_fn
        )
        default_jacobian = _make_default_jacobian(times, primary_param)

        # 3. Configure identifiability gate
        gate_options = None
        if request.identifiability_gate != "off":
            gate_options = IdentifiabilityGateOptions(
                policy=request.identifiability_gate,
                relative_tolerance=1e-6,
            )

        # 4. Solve MAP problem
        problem = MapEstimatorProblem(
            trajectory=trajectory,
            evaluation_times=times,
            initial_coefficients=initial_coefficients,
            shared_parameters=param_block,
            residual=default_residual,
            jacobian=default_jacobian,
            options=MapEstimatorOptions(
                max_iterations=request.max_iterations,
                non_finite_policy=request.non_finite_policy,
                identifiability=gate_options,
            ),
        )

        map_result = solve_single_trial_map(problem)

        # 5. Compute provenance hash
        prov_hasher = hashlib.sha256()
        prov_hasher.update(obs.source_hash.encode("utf-8"))
        prov_hasher.update(
            json.dumps(map_result.parameters, sort_keys=True).encode("utf-8")
        )
        prov_hasher.update(str(map_result.objective).encode("utf-8"))
        provenance_hash = prov_hasher.hexdigest()

        gate_dict = (
            map_result.identifiability.to_dict()
            if map_result.identifiability is not None
            else None
        )

        return EstimationJobResult(
            task_id=request.task_id,
            status=EstimationJobStatus.COMPLETED
            if map_result.success
            else EstimationJobStatus.FAILED,
            success=map_result.success,
            estimated_parameters=dict(map_result.parameters),
            identifiability_report=gate_dict,
            iterations=map_result.n_iterations,
            cost=float(map_result.objective),
            source_hash=obs.source_hash,
            provenance_hash=provenance_hash,
        )
