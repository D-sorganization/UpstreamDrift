"""Moving-horizon estimator for near-real-time canonical-core fitting.

The estimator keeps a bounded sample window, solves that window with fixed
parameters, and warm-starts each new window from the previously solved spline.
Residual and Jacobian callables match the CC-19 MAP surface so batch and
windowed modes can share the same engine residual kernels.
Extended in DIME-07 (#11428) with arrival information, rank-revealing Schur
marginalization, and safe window commits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from src.shared.python.contracts import PreconditionError, require
from src.shared.python.estimation.map_estimator import (
    CubicHermiteSplineTrajectory,
    MapDecisionLayout,
    MapEstimatorOptions,
    MapEstimatorProblem,
    MapEstimatorResult,
    SharedParameterBlock,
    SplineTrajectoryEvaluation,
    solve_single_trial_map,
)

FixedResidualFn = Callable[
    [SplineTrajectoryEvaluation, Mapping[str, float]], np.ndarray
]
FixedJacobianFn = Callable[
    [SplineTrajectoryEvaluation, Mapping[str, float], MapDecisionLayout],
    np.ndarray,
]
ResultCallback = Callable[["MovingHorizonResult"], None]


@dataclass(frozen=True)
class MovingHorizonOptions:
    """Configuration for bounded near-real-time window solves."""

    window_size: int
    step_size: int = 1
    latency_budget_ms: float = 50.0
    solver_options: MapEstimatorOptions = MapEstimatorOptions(max_iterations=20)

    def __post_init__(self) -> None:
        require(self.window_size >= 2, "window_size must be at least 2")
        require(self.step_size >= 1, "step_size must be positive")
        require(self.step_size <= self.window_size, "step_size must fit the window")
        require(self.latency_budget_ms > 0.0, "latency_budget_ms must be positive")


@dataclass(frozen=True)
class MovingHorizonProblem:
    """Problem definition for a fixed-parameter moving-horizon solve."""

    n_dof: int
    fixed_parameters: Mapping[str, float]
    residual: FixedResidualFn
    jacobian: FixedJacobianFn | None = None
    options: MovingHorizonOptions = MovingHorizonOptions(window_size=2)
    callback: ResultCallback | None = None

    def __post_init__(self) -> None:
        require(self.n_dof > 0, "n_dof must be positive")
        for name, value in self.fixed_parameters.items():
            require(str(name).strip() != "", "parameter names must be non-empty")
            require(np.isfinite(float(value)), f"{name} must be finite")


@dataclass(frozen=True)
class MovingHorizonResult:
    """Result of one deterministic moving-horizon window solve."""

    success: bool
    window_index: int
    sample_start: int
    sample_stop: int
    window_times: np.ndarray
    coefficients: np.ndarray
    initial_coefficients: np.ndarray
    parameters: dict[str, float]
    residual: np.ndarray
    objective: float
    n_iterations: int
    latency_ms: float
    latency_budget_ms: float
    warm_started: bool
    message: str

    @property
    def over_budget(self) -> bool:
        """Whether the solve exceeded the configured per-window budget."""
        return self.latency_ms > self.latency_budget_ms

    def callback_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable callback payload for realtime bridges."""
        return {
            "success": self.success,
            "window_index": self.window_index,
            "sample_start": self.sample_start,
            "sample_stop": self.sample_stop,
            "objective": self.objective,
            "n_iterations": self.n_iterations,
            "latency_ms": self.latency_ms,
            "latency_budget_ms": self.latency_budget_ms,
            "over_budget": self.over_budget,
            "warm_started": self.warm_started,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class FailureDiagnostic:
    """Structured record of an uncommitted, failed window solve."""

    window_index: int
    timestamp_s: float
    reason: str
    raw_residual_norm: float = 0.0
    iterations: int = 0


@dataclass(frozen=True)
class ArrivalInformation:
    """Arrival cost representing summarized past information up to the window boundary."""

    reference_coefficients: np.ndarray
    sqrt_information: np.ndarray
    rank: int
    nullspace_basis: np.ndarray | None = None
    linearization_timestamp_s: float = 0.0
    marginalized_up_to_sample: int = 0
    information_vector: np.ndarray | None = None
    scheme: str = "schur_marginalization"
    gauge_policy: str = "rank_revealing_zero_fill"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ref = np.asarray(self.reference_coefficients, dtype=float)
        sqrt_info = np.asarray(self.sqrt_information, dtype=float)
        require(ref.ndim == 1, "reference_coefficients must be a 1D vector")
        require(sqrt_info.ndim == 2, "sqrt_information must be a 2D matrix")
        require(
            sqrt_info.shape[1] == ref.size,
            "sqrt_information columns must match reference dimension",
        )
        require(self.rank >= 0, "rank must be non-negative")
        require(self.rank <= ref.size, "rank cannot exceed parameter dimension")
        if self.nullspace_basis is not None:
            ns = np.asarray(self.nullspace_basis, dtype=float)
            require(
                ns.shape[0] == ref.size,
                "nullspace basis rows must match reference dimension",
            )

    def compute_residual(self, current_coefficients: np.ndarray) -> np.ndarray:
        """Compute whitened residual vector R * (x - x_ref) + g."""
        delta = np.asarray(current_coefficients, dtype=float) - np.asarray(
            self.reference_coefficients, dtype=float
        )
        res = self.sqrt_information @ delta
        if self.information_vector is not None:
            res = res + np.asarray(self.information_vector, dtype=float)
        return res

    def compute_cost(self, current_coefficients: np.ndarray) -> float:
        """Compute scalar arrival cost 0.5 * ||r||^2."""
        res = self.compute_residual(current_coefficients)
        return 0.5 * float(np.sum(res**2))


def marginalize_arrival_schur(
    H_00: np.ndarray,
    H_01: np.ndarray,
    H_11: np.ndarray,
    g_0: np.ndarray | None = None,
    g_1: np.ndarray | None = None,
    *,
    singular_value_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray | None, int, np.ndarray | None]:
    """Rank-revealing Schur complement marginalization without artificial diagonal jitter."""
    h_00 = np.asarray(H_00, dtype=float)
    h_01 = np.asarray(H_01, dtype=float)
    h_11 = np.asarray(H_11, dtype=float)
    require(h_00.ndim == 2 and h_00.shape[0] == h_00.shape[1], "H_00 must be square")
    require(h_11.ndim == 2 and h_11.shape[0] == h_11.shape[1], "H_11 must be square")
    require(
        h_01.shape[0] == h_00.shape[0] and h_01.shape[1] == h_11.shape[0],
        "H_01 dimensions must match H_00 and H_11",
    )

    u_0, s_0, vt_0 = np.linalg.svd(h_00)
    max_s0 = float(np.max(s_0)) if s_0.size > 0 else 0.0
    tol_0 = max(singular_value_tol, singular_value_tol * max_s0)
    mask_0 = s_0 > tol_0

    h_00_pinv = np.zeros_like(h_00)
    if np.any(mask_0):
        inv_s = np.zeros_like(s_0)
        inv_s[mask_0] = 1.0 / s_0[mask_0]
        h_00_pinv = vt_0.T @ np.diag(inv_s) @ u_0.T

    h_10 = h_01.T
    schur_info = h_11 - h_10 @ h_00_pinv @ h_01
    schur_info = 0.5 * (schur_info + schur_info.T)

    _, s_1, vt_1 = np.linalg.svd(schur_info)
    max_s1 = float(np.max(s_1)) if s_1.size > 0 else 0.0
    tol_1 = max(singular_value_tol, singular_value_tol * max_s1)
    mask_1 = s_1 > tol_1
    rank = int(np.sum(mask_1))

    dim_1 = h_11.shape[0]
    sqrt_info = np.zeros((dim_1, dim_1), dtype=float)
    if rank > 0:
        sqrt_diag = np.sqrt(s_1[mask_1])
        sqrt_info[:rank, :] = np.diag(sqrt_diag) @ vt_1[mask_1, :]

    nullspace: np.ndarray | None = None
    if rank < dim_1:
        nullspace = vt_1[~mask_1, :].T

    g_out: np.ndarray | None = None
    if g_1 is not None and g_0 is not None:
        g_out = np.asarray(g_1, dtype=float) - h_10 @ h_00_pinv @ np.asarray(
            g_0, dtype=float
        )

    return sqrt_info, g_out, rank, nullspace


@dataclass
class _SampleBuffer:
    times: list[float] = field(default_factory=list)
    q_rows: list[np.ndarray] = field(default_factory=list)
    first_sample_index: int = 0

    @property
    def size(self) -> int:
        return len(self.times)

    def append(self, times: Sequence[float], q_samples: np.ndarray, n_dof: int) -> None:
        sample_times = np.asarray(times, dtype=float)
        sample_q = np.asarray(q_samples, dtype=float)
        require(sample_times.ndim == 1, "times must be a 1D array")
        require(sample_q.shape == (sample_times.size, n_dof), "q_samples shape invalid")
        require(bool(np.all(np.isfinite(sample_times))), "times must be finite")
        require(bool(np.all(np.isfinite(sample_q))), "q_samples must be finite")
        if self.times and sample_times.size:
            if sample_times[0] <= self.times[-1]:
                raise PreconditionError(
                    f"new samples must advance monotonically: {sample_times[0]} <= {self.times[-1]}"
                )
        if sample_times.size > 1:
            if bool(np.any(np.diff(sample_times) <= 0.0)):
                raise PreconditionError("times must increase strictly")
        self.times.extend(float(value) for value in sample_times)
        self.q_rows.extend(np.array(row, dtype=float) for row in sample_q)

    def clear_and_set(
        self, times: Sequence[float], q_samples: np.ndarray, n_dof: int
    ) -> None:
        """Clear existing buffer entries and populate fresh samples for recovery."""
        self.times.clear()
        self.q_rows.clear()
        self.append(times, q_samples, n_dof)

    def trim_to(self, max_size: int) -> None:
        overflow = self.size - max_size
        if overflow <= 0:
            return
        del self.times[:overflow]
        del self.q_rows[:overflow]
        self.first_sample_index += overflow

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.times, dtype=float), np.vstack(self.q_rows)


class MovingHorizonEstimator:
    """Bounded moving-horizon estimator with deterministic window advancement and safe commits."""

    def __init__(self, problem: MovingHorizonProblem) -> None:
        self._problem = problem
        self._buffer = _SampleBuffer()
        self._last_solved_stop = 0
        self._window_index = 0
        self._previous_trajectory: CubicHermiteSplineTrajectory | None = None
        self._previous_coefficients: np.ndarray | None = None
        self._last_accepted_coefficients: np.ndarray | None = None
        self._failure_diagnostics: list[FailureDiagnostic] = []
        self._arrival_information: ArrivalInformation | None = None

    @property
    def buffered_sample_count(self) -> int:
        """Number of retained samples in the bounded window buffer."""
        return self._buffer.size

    @property
    def last_accepted_coefficients(self) -> np.ndarray | None:
        """Coefficients from the most recent successful and committed window solve."""
        if self._last_accepted_coefficients is None:
            return None
        return np.array(self._last_accepted_coefficients, dtype=float)

    @property
    def failure_diagnostics(self) -> tuple[FailureDiagnostic, ...]:
        """Sequence of uncommitted failure diagnostics recorded during execution."""
        return tuple(self._failure_diagnostics)

    @property
    def arrival_information(self) -> ArrivalInformation | None:
        """Current arrival cost summarizing marginalized past trajectory information."""
        return self._arrival_information

    def update_fixed_parameters(self, parameters: Mapping[str, float]) -> None:
        """Update problem parameters with strict safeguard against mid-trajectory revision."""
        if self._last_solved_stop > 0:
            raise PreconditionError(
                "Parameter revision detected: cannot revise fixed parameters during an active "
                "trajectory without explicit arrival re-initialization"
            )
        self._problem = MovingHorizonProblem(
            n_dof=self._problem.n_dof,
            fixed_parameters=dict(parameters),
            residual=self._problem.residual,
            jacobian=self._problem.jacobian,
            options=self._problem.options,
            callback=self._problem.callback,
        )

    def append_samples(self, times: Sequence[float], q_samples: np.ndarray) -> None:
        """Append strictly increasing samples and retain only the active window."""
        self._buffer.append(times, q_samples, self._problem.n_dof)
        self._buffer.trim_to(self._problem.options.window_size)

    def ready(self) -> bool:
        """Return true when enough new samples exist for the next solve."""
        if self._buffer.size < self._problem.options.window_size:
            return False
        retained_stop = self._buffer.first_sample_index + self._buffer.size
        if self._last_solved_stop == 0:
            return True
        return retained_stop - self._last_solved_stop >= self._problem.options.step_size

    def build_current_problem(self) -> MapEstimatorProblem:
        """Build the fixed-parameter MAP problem for the retained window."""
        require(self.ready(), "not enough new samples for a moving-horizon solve")
        times, q_samples = self._buffer.arrays()
        trajectory = CubicHermiteSplineTrajectory(times, self._problem.n_dof)
        initial_coefficients = self._initial_coefficients(trajectory, times, q_samples)
        return self._map_problem(trajectory, times, initial_coefficients)

    def solve_next(
        self, max_boundary_jump: float | None = None
    ) -> MovingHorizonResult | None:
        """Solve the next ready window with safe failure handling and commit validation."""
        if not self.ready():
            return None
        map_problem = self.build_current_problem()
        warm_started = self._previous_coefficients is not None

        # Check boundary continuity jump against last accepted solution or within window
        if max_boundary_jump is not None:
            times, q_samples = self._buffer.arrays()
            diffs = np.abs(np.diff(q_samples, axis=0))
            max_sample_jump = float(np.max(diffs)) if diffs.size > 0 else 0.0
            knot_jump = 0.0
            if self._last_accepted_coefficients is not None:
                knot_jump = float(
                    np.max(
                        np.abs(
                            map_problem.initial_coefficients[: self._problem.n_dof]
                            - self._last_accepted_coefficients[: self._problem.n_dof]
                        )
                    )
                )
            jump = max(max_sample_jump, knot_jump)
            if jump > max_boundary_jump:
                diag = FailureDiagnostic(
                    window_index=self._window_index,
                    timestamp_s=float(map_problem.evaluation_times[0]),
                    reason=f"Window boundary jump {jump:.2f} exceeds threshold {max_boundary_jump:.2f}",
                    raw_residual_norm=jump,
                    iterations=0,
                )
                self._failure_diagnostics.append(diag)
                failed_res = MapEstimatorResult(
                    success=False,
                    coefficients=map_problem.initial_coefficients,
                    parameters=dict(self._problem.fixed_parameters),
                    residual=np.array([jump]),
                    objective=0.5 * jump**2,
                    n_iterations=0,
                    message=f"Window boundary jump {jump:.2f} exceeds threshold {max_boundary_jump:.2f}",
                )
                result = self._to_result(map_problem, failed_res, 0.0, warm_started)
                if self._problem.callback is not None:
                    self._problem.callback(result)
                return result

        started = perf_counter()
        map_result = solve_single_trial_map(map_problem)
        latency_ms = (perf_counter() - started) * 1000.0

        is_sentinel_failed = bool(
            map_result.objective >= 1e8 or np.any(np.abs(map_result.residual) >= 1e11)
        )
        is_finite = bool(
            np.all(np.isfinite(map_result.coefficients))
            and np.all(np.isfinite(map_result.residual))
            and not is_sentinel_failed
        )
        if not map_result.success or not is_finite:
            reason = (
                "Residual hit non-finite sentinel"
                if is_sentinel_failed
                else (
                    map_result.message or "Solver failed or produced non-finite output"
                )
            )
            diag = FailureDiagnostic(
                window_index=self._window_index,
                timestamp_s=float(map_problem.evaluation_times[0]),
                reason=reason,
                raw_residual_norm=(
                    float(np.linalg.norm(map_result.residual)) if is_finite else np.nan
                ),
                iterations=map_result.n_iterations,
            )
            self._failure_diagnostics.append(diag)
            failed_map_result = MapEstimatorResult(
                success=False,
                coefficients=map_result.coefficients,
                parameters=map_result.parameters,
                residual=map_result.residual,
                objective=map_result.objective,
                n_iterations=map_result.n_iterations,
                message=reason,
            )
            result = self._to_result(
                map_problem, failed_map_result, latency_ms, warm_started
            )
            if self._problem.callback is not None:
                self._problem.callback(result)
            return result

        # Successful solve: safely commit state advancement
        result = self._to_result(map_problem, map_result, latency_ms, warm_started)
        self._last_accepted_coefficients = map_result.coefficients
        self._previous_trajectory = map_problem.trajectory
        self._previous_coefficients = map_result.coefficients
        self._last_solved_stop = self._buffer.first_sample_index + self._buffer.size
        self._window_index += 1
        if self._problem.callback is not None:
            self._problem.callback(result)
        return result

    def recover_and_solve_next(
        self, times: Sequence[float], q_samples: np.ndarray
    ) -> MovingHorizonResult | None:
        """Reset sample buffer with valid recovery observations and solve next window."""
        self._buffer.clear_and_set(times, q_samples, self._problem.n_dof)
        return self.solve_next()

    def _initial_coefficients(
        self,
        trajectory: CubicHermiteSplineTrajectory,
        times: np.ndarray,
        q_samples: np.ndarray,
    ) -> np.ndarray:
        seed_coeffs = (
            self._last_accepted_coefficients
            if self._last_accepted_coefficients is not None
            else self._previous_coefficients
        )
        if self._previous_trajectory is None or seed_coeffs is None:
            return trajectory.initial_coefficients_from_samples(times, q_samples)
        previous = self._previous_trajectory.evaluate(seed_coeffs, times)
        return trajectory.pack(previous.q, previous.v)

    def _map_problem(
        self,
        trajectory: CubicHermiteSplineTrajectory,
        times: np.ndarray,
        initial_coefficients: np.ndarray,
    ) -> MapEstimatorProblem:
        fixed = dict(self._problem.fixed_parameters)

        def residual(evaluation: SplineTrajectoryEvaluation, _parameters) -> np.ndarray:
            return self._problem.residual(evaluation, fixed)

        jacobian_wrapper = None
        problem_jacobian = self._problem.jacobian
        if problem_jacobian is not None:

            def jacobian_wrapper(evaluation, _parameters, layout):
                return problem_jacobian(evaluation, fixed, layout)

        return MapEstimatorProblem(
            trajectory=trajectory,
            evaluation_times=times,
            initial_coefficients=initial_coefficients,
            shared_parameters=SharedParameterBlock.from_specs([]),
            residual=residual,
            jacobian=jacobian_wrapper,
            options=self._problem.options.solver_options,
        )

    def _to_result(
        self,
        problem: MapEstimatorProblem,
        map_result: MapEstimatorResult,
        latency_ms: float,
        warm_started: bool,
    ) -> MovingHorizonResult:
        sample_start = self._buffer.first_sample_index
        sample_stop = sample_start + self._buffer.size
        return MovingHorizonResult(
            success=map_result.success,
            window_index=self._window_index,
            sample_start=sample_start,
            sample_stop=sample_stop,
            window_times=np.array(problem.evaluation_times, dtype=float),
            coefficients=map_result.coefficients,
            initial_coefficients=np.array(problem.initial_coefficients, dtype=float),
            parameters=dict(self._problem.fixed_parameters),
            residual=map_result.residual,
            objective=map_result.objective,
            n_iterations=map_result.n_iterations,
            latency_ms=latency_ms,
            latency_budget_ms=self._problem.options.latency_budget_ms,
            warm_started=warm_started,
            message=map_result.message,
        )
