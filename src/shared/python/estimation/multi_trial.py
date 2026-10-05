"""Multi-trial MAP stacking with one shared parameter block.

This module extends the CC-19 single-trial estimator without taking ownership
of engine residual math. Each trial or view contributes its own spline
trajectory block while all observations see the same shared parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import least_squares

from src.shared.python.contracts import require
from src.shared.python.estimation.map_estimator import (
    CubicHermiteSplineTrajectory,
    MapEstimatorOptions,
    NonFiniteCounter,
    SharedParameterBlock,
    SplineTrajectoryEvaluation,
    _require_times_within_knot_span,
    _sentinel_for_non_finite,
    _warn_if_sentinel_fired,
)
from src.shared.python.simulation_backends.provenance import ProvenanceStamp

if TYPE_CHECKING:
    from src.shared.python.estimation.identifiability import (
        IdentifiabilityGateReport,
    )

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
CovarianceStatus = Literal["estimated", "rank_deficient"]


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
    n_non_finite_evaluations: int = 0
    identifiability: IdentifiabilityGateReport | None = None
    locked_by_gate: tuple[str, ...] = field(default_factory=tuple)
    covariance_status: CovarianceStatus = "estimated"

    def posterior_variance(self, name: str) -> float:
        """Return the approximate posterior variance for an unlocked parameter.

        ``NaN`` when ``covariance_status == "rank_deficient"``: the marginal is
        undefined and is reported as unavailable rather than approximated.
        """
        try:
            index = self.posterior_parameter_names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return float(self.posterior_covariance[index, index])


def solve_multi_trial_map(problem: MultiTrialMapProblem) -> MultiTrialMapResult:
    """Solve a stacked MAP problem with trial-local trajectories and shared theta."""
    _validate_problem(problem)
    gate_report, locked = _apply_identifiability_gate(problem)
    if locked:
        problem = _with_locked_parameters(problem, locked)
    layout = _build_layout(problem)
    x0 = _pack_decision(problem)
    lower, upper = _decision_bounds(problem)
    counter = NonFiniteCounter()

    def residual_for_solver(x: np.ndarray) -> np.ndarray:
        return _objective_residual(problem, layout, x, counter)

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
    covariance, covariance_status = _posterior_covariance(
        problem, layout, result.x, residual.size
    )
    _warn_if_sentinel_fired(counter, _all_jacobians_available(problem))
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
        n_non_finite_evaluations=counter.evaluations,
        identifiability=gate_report,
        locked_by_gate=tuple(locked),
        covariance_status=covariance_status,
    )


def _apply_identifiability_gate(
    problem: MultiTrialMapProblem,
) -> tuple[IdentifiabilityGateReport | None, list[str]]:
    """Run the #9758 gate on the free shared parameters at the initial guess."""
    from src.shared.python.estimation.identifiability import (
        IdentifiabilityGateOptions,
        gate_shared_parameters,
    )

    options = problem.options.identifiability or IdentifiabilityGateOptions()
    block = problem.shared_parameters
    if options.policy == "off" or block.free_size == 0:
        return None, []
    evaluations = [
        (
            observation,
            observation.trajectory.evaluate(
                np.asarray(observation.initial_coefficients, dtype=float),
                observation.evaluation_times,
            ),
        )
        for observation in problem.observations
    ]

    def residual_of_free(free_values: np.ndarray) -> np.ndarray:
        parameters = block.to_mapping(block.expand_free_vector(free_values))
        return np.concatenate(
            [
                np.asarray(
                    observation.residual(observation, evaluation, parameters),
                    dtype=float,
                ).reshape(-1)
                for observation, evaluation in evaluations
            ]
        )

    report = gate_shared_parameters(residual_of_free, block, options)
    return report, list(report.locked_parameters)


def _with_locked_parameters(
    problem: MultiTrialMapProblem, names: Sequence[str]
) -> MultiTrialMapProblem:
    specs = tuple(
        replace(spec, locked=True) if spec.name in names else spec
        for spec in problem.shared_parameters.specs
    )
    return replace(problem, shared_parameters=SharedParameterBlock.from_specs(specs))


def stack_shared_parameter_jacobians(
    rows_by_observation: Sequence[np.ndarray],
) -> np.ndarray:
    """Stack per-trial shared-parameter Jacobians for identifiability checks."""
    if not rows_by_observation:
        raise ValueError("at least one Jacobian block is required")
    rows = [np.asarray(row, dtype=float) for row in rows_by_observation]
    if rows[0].ndim != 2:
        raise ValueError("Jacobian blocks must be 2D with matching widths")
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
    require(bool(np.all(np.isfinite(matrix))), "jacobian must be finite")
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
    require(bool(np.all(np.isfinite(times))), "evaluation_times must be finite")
    _require_times_within_knot_span(times, observation.trajectory.knot_times)
    coeffs = np.asarray(observation.initial_coefficients, dtype=float)
    require(
        coeffs.shape == (observation.trajectory.coefficient_size,),
        "initial_coefficients shape must match trajectory",
    )
    require(bool(np.all(np.isfinite(coeffs))), "initial_coefficients must be finite")


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
    counter: NonFiniteCounter | None = None,
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
        residual = _sentinel_for_non_finite(
            residual, policy=problem.options.non_finite_policy, counter=counter
        )
        residuals.append(residual)
    residuals.append(_prior_row_layout(problem.shared_parameters).residual(free_values))
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
    rows.append(_prior_row_layout(problem.shared_parameters).jacobian(layout))
    return np.vstack(rows)


@dataclass(frozen=True)
class _PriorRowLayout:
    """Row layout of the Gaussian prior block, shared by residual and Jacobian.

    One row per **free** parameter that carries both ``prior`` and
    ``prior_scale``, in free-parameter order. A locked parameter is not a
    decision variable: its prior residual would be a constant row with a zero
    Jacobian row, which shifts the objective by a constant, leaves its gradient
    and Gauss-Newton Fisher matrix unchanged, and therefore cannot affect the
    MAP estimate or the posterior. Such rows are excluded from **both** the
    residual and the Jacobian, matching
    :meth:`SharedParameterBlock.free_prior_residuals` used by the single-trial
    estimator (#11548). Building both blocks from this one object is what
    keeps their row counts and order identical.
    """

    free_indices: np.ndarray
    priors: np.ndarray
    scales: np.ndarray

    @property
    def n_rows(self) -> int:
        """Number of prior rows."""
        return int(self.free_indices.size)

    def residual(self, free_values: np.ndarray) -> np.ndarray:
        """Prior residuals ``(theta_free[i] - prior_i) / scale_i``.

        Precondition: ``free_values`` is a finite 1-D vector covering every
        indexed free parameter. Postcondition: shape ``(n_rows,)``.
        """
        values = np.asarray(free_values, dtype=float)
        require(values.ndim == 1, "free_values must be 1D")
        require(
            self.n_rows == 0 or int(self.free_indices.max()) < values.size,
            "free_values does not cover the prior rows",
        )
        rows = (values[self.free_indices] - self.priors) / self.scales
        assert rows.shape == (self.n_rows,)
        return rows

    def jacobian(self, layout: MultiTrialDecisionLayout) -> np.ndarray:
        """Jacobian of :meth:`residual` over the full decision vector.

        Postcondition: shape ``(n_rows, layout.size)`` with ``1 / scale_i`` in
        the decision column of each prior's free parameter.
        """
        rows = np.zeros((self.n_rows, layout.size), dtype=float)
        columns = layout.trajectory_size + self.free_indices
        rows[np.arange(self.n_rows), columns] = 1.0 / self.scales
        return rows


def _prior_row_layout(parameter_block: SharedParameterBlock) -> _PriorRowLayout:
    """Return the single prior row layout used by residual and Jacobian."""
    indexed = [
        (index, float(spec.prior), float(spec.prior_scale))
        for index, spec in enumerate(parameter_block.free_specs)
        if spec.prior is not None and spec.prior_scale is not None
    ]
    return _PriorRowLayout(
        free_indices=np.array([item[0] for item in indexed], dtype=int),
        priors=np.array([item[1] for item in indexed], dtype=float),
        scales=np.array([item[2] for item in indexed], dtype=float),
    )


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
    n_residual_rows: int,
) -> tuple[np.ndarray, CovarianceStatus]:
    """Marginal Laplace covariance of the free shared parameters.

    Builds the objective Jacobian at ``decision`` and delegates to
    :func:`_marginal_shared_block`, which documents the calculation and its
    rank-deficiency behaviour.

    Preconditions: ``decision`` is finite with width ``layout.size``. Invariant:
    the Jacobian has exactly ``n_residual_rows`` rows (one shared row layout).
    Postcondition: a square matrix of side ``free_size`` and its status.
    """
    if problem.shared_parameters.free_size == 0:
        return np.zeros((0, 0), dtype=float), "estimated"
    require(decision.shape == (layout.size,), "decision width must match layout")
    require(bool(np.all(np.isfinite(decision))), "decision must be finite")
    if _all_jacobians_available(problem):
        jacobian = _objective_jacobian(problem, layout, decision)
    else:
        jacobian = _finite_difference_jacobian(
            lambda x: _objective_residual(problem, layout, x),
            decision,
        )
    require(jacobian.shape[1] == layout.size, "jacobian width must match layout")
    assert jacobian.shape[0] == n_residual_rows, "residual/Jacobian row mismatch"
    return _marginal_shared_block(
        jacobian,
        trajectory_size=layout.trajectory_size,
        regularization=problem.covariance_regularization,
    )


def _marginal_shared_block(
    jacobian: np.ndarray,
    trajectory_size: int,
    regularization: float,
) -> tuple[np.ndarray, CovarianceStatus]:
    """Shared-parameter block of the inverse regularised Fisher matrix.

    The Gauss-Newton Fisher matrix ``F = J^T J + lambda I`` spans every
    decision column: the per-trial trajectory coefficients ``t`` (the first
    ``trajectory_size`` columns) and the free shared parameters ``s``. The
    trajectory coefficients are nuisance parameters, so the reported
    covariance is the **marginal** over ``s``, the ``s`` block of ``F^-1``,
    which equals the inverse Schur complement
    ``(F_ss - F_st F_tt^-1 F_ts)^-1``. Inverting ``F_ss`` alone would be the
    covariance *conditional* on the trajectories and understates uncertainty
    whenever they are correlated with ``s`` (#11548).

    Rank deficiency fails closed. ``F`` is treated as singular when its
    numerical rank, counting singular values above
    ``sigma_max(F) * n * eps`` (``n`` = number of columns; the
    :func:`numpy.linalg.matrix_rank` default), is below ``n``. A singular
    ``F`` has no inverse, the Schur complement is singular, and the marginal
    variance of at least one confounded direction is unbounded; the
    Moore-Penrose block is *not* that marginal (for ``J = [1, 1]`` it gives
    ``0.25`` where the marginal is infinite). The block is then returned
    filled with ``NaN`` and status ``"rank_deficient"``, the same convention
    :mod:`.fit_uncertainty` uses. A positive ``lambda`` large enough to clear
    the tolerance makes ``F`` invertible; the result is then the exact
    marginal of the *regularised* (prior-augmented) problem.

    Preconditions: ``jacobian`` is finite 2-D with more than
    ``trajectory_size`` columns; ``regularization >= 0``. Postcondition: a
    square matrix of side ``n - trajectory_size`` and its status.
    """
    matrix = np.asarray(jacobian, dtype=float)
    require(matrix.ndim == 2, "jacobian must be 2D")
    require(0 <= trajectory_size < matrix.shape[1], "trajectory_size out of range")
    require(bool(np.all(np.isfinite(matrix))), "jacobian must be finite")
    require(regularization >= 0.0, "regularization must be non-negative")
    n_columns = matrix.shape[1]
    shared = slice(trajectory_size, n_columns)
    side = n_columns - trajectory_size
    fisher = matrix.T @ matrix + regularization * np.eye(n_columns)
    if np.linalg.matrix_rank(fisher, hermitian=True) < n_columns:
        return np.full((side, side), np.nan, dtype=float), "rank_deficient"
    full = shared_parameter_covariance(matrix, regularization=regularization)
    marginal = full[shared, shared].copy()
    assert marginal.shape == (side, side)
    return marginal, "estimated"


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
