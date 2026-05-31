"""Multi-trial MAP stacking with one shared parameter block.

This module extends the CC-19 single-trial estimator without taking ownership
of engine residual math. Each trial or view contributes its own spline
trajectory block while all observations see the same shared parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.shared.python.contracts import require
from src.shared.python.estimation.map_estimator import (
    CubicHermiteSplineTrajectory,
    MapEstimatorOptions,
    SharedParameterBlock,
    SplineTrajectoryEvaluation,
)
from src.shared.python.simulation_backends.provenance import ProvenanceStamp

MultiTrialResidualFn = Callable[
    ["MultiTrialObservation", SplineTrajectoryEvaluation, Mapping[str, float]],
    np.ndarray,
]
MultiTrialJacobianFn = Callable[
    [
        "MultiTrialObservation",
        SplineTrajectoryEvaluation,
        Mapping[str, float],
        "MultiTrialDecisionLayout",
    ],
    np.ndarray,
]


@dataclass(frozen=True)
class MultiTrialObservation:
    """One trial/view contribution to a shared-parameter MAP solve."""

    trial_id: str
    trajectory: CubicHermiteSplineTrajectory
    evaluation_times: np.ndarray
    initial_coefficients: np.ndarray
    residual: MultiTrialResidualFn
    jacobian: MultiTrialJacobianFn | None = None
    view_id: str | None = None

    @property
    def key(self) -> str:
        """Stable result key for this trial/view."""
        if self.view_id is None:
            return self.trial_id
        return f"{self.trial_id}:{self.view_id}"


@dataclass(frozen=True)
class _TrialSlice:
    key: str
    start: int
    stop: int


@dataclass(frozen=True)
class MultiTrialDecisionLayout:
    """Column layout for stacked trial trajectories plus free shared params."""

    trial_slices: tuple[_TrialSlice, ...]
    free_parameter_names: tuple[str, ...]

    @property
    def trajectory_size(self) -> int:
        """Total width occupied by all per-trial trajectory blocks."""
        if not self.trial_slices:
            return 0
        return self.trial_slices[-1].stop

    @property
    def size(self) -> int:
        """Total decision-vector width."""
        return self.trajectory_size + len(self.free_parameter_names)

    def trajectory_slice(self, trial_key: str) -> slice:
        """Return the absolute decision-vector slice for a trial/view key."""
        for item in self.trial_slices:
            if item.key == trial_key:
                return slice(item.start, item.stop)
        raise KeyError(trial_key)

    def parameter_column(self, name: str) -> int:
        """Return the absolute decision-vector column for an unlocked parameter."""
        try:
            index = self.free_parameter_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.trajectory_size + index


@dataclass(frozen=True)
class MultiTrialMapProblem:
    """Complete multi-trial MAP problem with one shared parameter block."""

    observations: tuple[MultiTrialObservation, ...]
    shared_parameters: SharedParameterBlock
    options: MapEstimatorOptions = MapEstimatorOptions()
    covariance_regularization: float = 1e-12
    provenance: ProvenanceStamp | None = None


@dataclass(frozen=True)
class MultiTrialMapResult:
    """Deterministic result of a multi-trial shared-parameter MAP solve."""

    success: bool
    coefficients_by_trial: dict[str, np.ndarray]
    parameters: dict[str, float]
    posterior_parameter_names: tuple[str, ...]
    posterior_covariance: np.ndarray
    residual: np.ndarray
    objective: float
    n_iterations: int
    message: str
    provenance: ProvenanceStamp | None = None

    def posterior_variance(self, name: str) -> float:
        """Return the approximate posterior variance for an unlocked parameter."""
        try:
            index = self.posterior_parameter_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return float(self.posterior_covariance[index, index])


def solve_multi_trial_map(problem: MultiTrialMapProblem) -> MultiTrialMapResult:
    """Solve a stacked MAP problem with trial-local trajectories and shared theta."""
    _validate_problem(problem)
    layout = _build_layout(problem)
    x0 = _pack_decision(problem)
    lower, upper = _decision_bounds(problem)

    def residual_for_solver(x: np.ndarray) -> np.ndarray:
        return _objective_residual(problem, layout, x)

    jacobian_for_solver = None
    if _all_jacobians_available(problem):

        def jacobian_for_solver(x: np.ndarray) -> np.ndarray:
            return _objective_jacobian(problem, layout, x)

    method = problem.options.method
    if method == "lm" and _has_finite_bounds(lower, upper):
        method = "trf"
    result = least_squares(
        residual_for_solver,
        x0,
        jac=jacobian_for_solver if jacobian_for_solver is not None else "2-point",
        bounds=(lower, upper),
        method=method,
        max_nfev=problem.options.max_iterations,
        xtol=problem.options.xtol,
        ftol=problem.options.ftol,
        gtol=problem.options.gtol,
    )
    residual = residual_for_solver(result.x)
    coefficients = _unpack_coefficients(problem, layout, result.x)
    free_values = result.x[layout.trajectory_size :]
    full_values = problem.shared_parameters.expand_free_vector(free_values)
    parameters = problem.shared_parameters.to_mapping(full_values)
    covariance = _posterior_covariance(problem, layout, result.x)
    return MultiTrialMapResult(
        success=bool(result.success),
        coefficients_by_trial=coefficients,
        parameters=parameters,
        posterior_parameter_names=problem.shared_parameters.free_parameter_names,
        posterior_covariance=covariance,
        residual=residual,
        objective=0.5 * float(np.vdot(residual, residual)),
        n_iterations=int(result.nfev),
        message=str(result.message),
        provenance=problem.provenance,
    )


def stack_shared_parameter_jacobians(
    rows_by_observation: Sequence[np.ndarray],
) -> np.ndarray:
    """Stack per-trial shared-parameter Jacobians for identifiability checks."""
    if not rows_by_observation:
        raise ValueError("at least one Jacobian block is required")
    rows = [np.asarray(row, dtype=float) for row in rows_by_observation]
    width = rows[0].shape[1]
    for row in rows:
        if row.ndim != 2 or row.shape[1] != width:
            raise ValueError("Jacobian blocks must be 2D with matching widths")
        if not np.all(np.isfinite(row)):
            raise ValueError("Jacobian blocks must be finite")
    return np.vstack(rows)


def shared_parameter_covariance(
    jacobian: np.ndarray,
    noise_variance: float = 1.0,
    regularization: float = 1e-12,
) -> np.ndarray:
    """Approximate shared-parameter covariance from stacked residual Jacobians."""
    matrix = np.asarray(jacobian, dtype=float)
    require(matrix.ndim == 2, "jacobian must be 2D")
    require(matrix.shape[1] > 0, "jacobian must have at least one column")
    require(np.all(np.isfinite(matrix)), "jacobian must be finite")
    require(noise_variance > 0.0, "noise_variance must be positive")
    require(regularization >= 0.0, "regularization must be non-negative")
    fisher = matrix.T @ matrix
    if regularization:
        fisher = fisher + regularization * np.eye(matrix.shape[1])
    return float(noise_variance) * np.linalg.pinv(fisher)


def _validate_problem(problem: MultiTrialMapProblem) -> None:
    require(len(problem.observations) > 0, "at least one observation is required")
    keys = [observation.key for observation in problem.observations]
    require(len(keys) == len(set(keys)), "trial/view keys must be unique")
    require(problem.options.max_iterations > 0, "max_iterations must be positive")
    require(
        problem.covariance_regularization >= 0.0,
        "covariance_regularization must be non-negative",
    )
    for observation in problem.observations:
        _validate_observation(observation)


def _validate_observation(observation: MultiTrialObservation) -> None:
    require(bool(observation.trial_id.strip()), "trial_id must be non-empty")
    times = np.asarray(observation.evaluation_times, dtype=float)
    require(times.ndim == 1, "evaluation_times must be a 1D array")
    require(np.all(np.isfinite(times)), "evaluation_times must be finite")
    coeffs = np.asarray(observation.initial_coefficients, dtype=float)
    require(
        coeffs.shape == (observation.trajectory.coefficient_size,),
        "initial_coefficients shape must match trajectory",
    )
    require(np.all(np.isfinite(coeffs)), "initial_coefficients must be finite")


def _build_layout(problem: MultiTrialMapProblem) -> MultiTrialDecisionLayout:
    trial_slices = []
    offset = 0
    for observation in problem.observations:
        width = observation.trajectory.coefficient_size
        trial_slices.append(_TrialSlice(observation.key, offset, offset + width))
        offset += width
    return MultiTrialDecisionLayout(
        trial_slices=tuple(trial_slices),
        free_parameter_names=problem.shared_parameters.free_parameter_names,
    )


def _pack_decision(problem: MultiTrialMapProblem) -> np.ndarray:
    coefficients = [
        np.asarray(observation.initial_coefficients, dtype=float)
        for observation in problem.observations
    ]
    coefficients.append(problem.shared_parameters.free_initial_vector())
    return np.concatenate(coefficients)


def _decision_bounds(problem: MultiTrialMapProblem) -> tuple[np.ndarray, np.ndarray]:
    trajectory_size = sum(
        observation.trajectory.coefficient_size for observation in problem.observations
    )
    param_lower, param_upper = problem.shared_parameters.free_bounds()
    lower = np.concatenate([np.full(trajectory_size, -np.inf), param_lower])
    upper = np.concatenate([np.full(trajectory_size, np.inf), param_upper])
    return lower, upper


def _objective_residual(
    problem: MultiTrialMapProblem,
    layout: MultiTrialDecisionLayout,
    decision: np.ndarray,
) -> np.ndarray:
    free_values = decision[layout.trajectory_size :]
    parameter_values = problem.shared_parameters.expand_free_vector(free_values)
    parameters = problem.shared_parameters.to_mapping(parameter_values)
    residuals = []
    for observation in problem.observations:
        coefficients = decision[layout.trajectory_slice(observation.key)]
        evaluation = observation.trajectory.evaluate(
            coefficients,
            observation.evaluation_times,
        )
        residual = np.asarray(
            observation.residual(observation, evaluation, parameters),
            dtype=float,
        )
        if residual.ndim != 1:
            raise ValueError("residual callable must return a 1D array")
        if not np.all(np.isfinite(residual)):
            raise ValueError("residual callable returned non-finite values")
        residuals.append(residual)
    residuals.append(problem.shared_parameters.prior_residuals(parameter_values))
    return np.concatenate(residuals)


def _objective_jacobian(
    problem: MultiTrialMapProblem,
    layout: MultiTrialDecisionLayout,
    decision: np.ndarray,
) -> np.ndarray:
    free_values = decision[layout.trajectory_size :]
    parameter_values = problem.shared_parameters.expand_free_vector(free_values)
    parameters = problem.shared_parameters.to_mapping(parameter_values)
    rows = []
    for observation in problem.observations:
        if observation.jacobian is None:
            raise ValueError("jacobian callable must be provided")
        coefficients = decision[layout.trajectory_slice(observation.key)]
        evaluation = observation.trajectory.evaluate(
            coefficients,
            observation.evaluation_times,
        )
        jacobian = np.asarray(
            observation.jacobian(observation, evaluation, parameters, layout),
            dtype=float,
        )
        if jacobian.ndim != 2 or jacobian.shape[1] != layout.size:
            raise ValueError(f"jacobian callable must return (*, {layout.size})")
        rows.append(jacobian)
    prior = _free_prior_jacobian(problem.shared_parameters, layout)
    if prior.shape[0]:
        rows.append(prior)
    return np.vstack(rows)


def _free_prior_jacobian(
    parameter_block: SharedParameterBlock,
    layout: MultiTrialDecisionLayout,
) -> np.ndarray:
    rows = []
    for spec in parameter_block.free_specs:
        if spec.prior is None or spec.prior_scale is None:
            continue
        row = np.zeros(layout.size, dtype=float)
        row[layout.parameter_column(spec.name)] = 1.0 / spec.prior_scale
        rows.append(row)
    if not rows:
        return np.zeros((0, layout.size), dtype=float)
    return np.vstack(rows)


def _unpack_coefficients(
    problem: MultiTrialMapProblem,
    layout: MultiTrialDecisionLayout,
    decision: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        observation.key: decision[layout.trajectory_slice(observation.key)].copy()
        for observation in problem.observations
    }


def _posterior_covariance(
    problem: MultiTrialMapProblem,
    layout: MultiTrialDecisionLayout,
    decision: np.ndarray,
) -> np.ndarray:
    if problem.shared_parameters.free_size == 0:
        return np.zeros((0, 0), dtype=float)
    if _all_jacobians_available(problem):
        jacobian = _objective_jacobian(problem, layout, decision)
    else:
        jacobian = _finite_difference_jacobian(
            lambda x: _objective_residual(problem, layout, x),
            decision,
        )
    shared_jacobian = jacobian[:, layout.trajectory_size :]
    return shared_parameter_covariance(
        shared_jacobian,
        regularization=problem.covariance_regularization,
    )


def _finite_difference_jacobian(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    decision: np.ndarray,
) -> np.ndarray:
    base = residual_fn(decision)
    jacobian = np.zeros((base.size, decision.size), dtype=float)
    step = np.sqrt(np.finfo(float).eps)
    for column in range(decision.size):
        delta = np.zeros(decision.size, dtype=float)
        delta[column] = step * max(1.0, abs(float(decision[column])))
        jacobian[:, column] = (residual_fn(decision + delta) - base) / delta[column]
    return jacobian


def _all_jacobians_available(problem: MultiTrialMapProblem) -> bool:
    return all(observation.jacobian is not None for observation in problem.observations)


def _has_finite_bounds(lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)))
