"""Moving-horizon estimator for near-real-time canonical-core fitting.

The estimator keeps a bounded sample window, solves that window with fixed
parameters, and warm-starts each new window from the previously solved spline.
Residual and Jacobian callables match the CC-19 MAP surface so batch and
windowed modes can share the same engine residual kernels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from src.shared.python.contracts import require
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
            require(
                sample_times[0] > self.times[-1],
                "new samples must advance monotonically",
            )
        if sample_times.size > 1:
            require(bool(np.all(np.diff(sample_times) > 0.0)), "times must increase")
        self.times.extend(float(value) for value in sample_times)
        self.q_rows.extend(np.array(row, dtype=float) for row in sample_q)

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
    """Bounded moving-horizon estimator with deterministic window advancement."""

    def __init__(self, problem: MovingHorizonProblem) -> None:
        self._problem = problem
        self._buffer = _SampleBuffer()
        self._last_solved_stop = 0
        self._window_index = 0
        self._previous_trajectory: CubicHermiteSplineTrajectory | None = None
        self._previous_coefficients: np.ndarray | None = None

    @property
    def buffered_sample_count(self) -> int:
        """Number of retained samples in the bounded window buffer."""
        return self._buffer.size

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

    def solve_next(self) -> MovingHorizonResult | None:
        """Solve the next ready window, or return ``None`` if no window advanced."""
        if not self.ready():
            return None
        map_problem = self.build_current_problem()
        warm_started = self._previous_coefficients is not None
        started = perf_counter()
        map_result = solve_single_trial_map(map_problem)
        latency_ms = (perf_counter() - started) * 1000.0
        result = self._to_result(map_problem, map_result, latency_ms, warm_started)
        self._previous_trajectory = map_problem.trajectory
        self._previous_coefficients = map_result.coefficients
        self._last_solved_stop = self._buffer.first_sample_index + self._buffer.size
        self._window_index += 1
        if self._problem.callback is not None:
            self._problem.callback(result)
        return result

    def _initial_coefficients(
        self,
        trajectory: CubicHermiteSplineTrajectory,
        times: np.ndarray,
        q_samples: np.ndarray,
    ) -> np.ndarray:
        if self._previous_trajectory is None or self._previous_coefficients is None:
            return trajectory.initial_coefficients_from_samples(times, q_samples)
        previous = self._previous_trajectory.evaluate(
            self._previous_coefficients, times
        )
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
