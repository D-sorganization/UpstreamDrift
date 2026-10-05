"""Outer Levenberg-Marquardt solver over kinematics, geometry and inertia (MOSAIC).

Outer variables ``xi = (C_1..K, s, theta)``: per-trial spline coefficients,
shared geometry and shared inertial parameters in *unconstrained consistency
coordinates* ``pi = pi(theta)`` (log-Cholesky / planar log coordinates), so
every iterate is a physically realisable body.  Inner variables (inputs, torque
template) are eliminated exactly by variable projection, giving the reduced
objective

    F(xi) = |r_obs|^2 + min_u |A(xi) u - b(xi)|^2 + |r_geom|^2 + |r_pi|^2 + |r_anchor|^2.

Each iteration assembles the reduced Gauss-Newton system (Schur complement,
:func:`reduced_gauss_newton`) and takes a Levenberg-Marquardt step with Nielsen
damping updates.  A continuation schedule on the dynamics weight moves the
solve from a pure kinematic fit (``factor = 0``) to the fully weighted
dynamics-consistent fit, which is the homotopy that makes convergence from an
IK seed reliable.

Dynamics Jacobians: w.r.t. inertia they are *exact* (``W Y dpi/dtheta``, the
regressor is linear in ``pi``); w.r.t. kinematics and geometry they are
vectorized central differences over all nodes at once (``2 (3 n_dof + n_geom)``
batched regressor evaluations per iteration).  An engine with analytic
derivatives can replace :func:`dynamics_jacobian_by_finite_differences`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require
from src.shared.python.estimation.mosaic.inertial import InertialParameterization
from src.shared.python.estimation.mosaic.inner_solve import (
    InnerProblem,
    InnerSolution,
    LinearAnchor,
    reduced_gauss_newton,
    solve_inner,
)
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
_DAMPING_CEILING = 1e12
_DAMPING_FLOOR = 1e-15
_MIN_DIAGONAL = 1e-12


class RegressorModel(Protocol):
    """Dynamics/kinematics surface the outer solver needs from any engine."""

    @property
    def n_dof(self) -> int: ...

    @property
    def n_parameters(self) -> int: ...

    @property
    def input_matrix(self) -> FloatArray: ...

    def regressor(self, q: FloatArray, v: FloatArray, a: FloatArray) -> FloatArray: ...

    def marker_positions(self, q: FloatArray, markers: Any) -> FloatArray: ...

    def marker_jacobians(
        self, q: FloatArray, markers: Any
    ) -> tuple[FloatArray, FloatArray]: ...


ModelFactory = Callable[[FloatArray], RegressorModel]


@dataclass(frozen=True)
class TrialData:
    """Observations of one trial on a shared kinematic basis."""

    times: FloatArray
    marker_positions: FloatArray
    markers: Any
    marker_sigma: float
    basis: BSplineBasis
    phase_index: IntArray | None

    def __post_init__(self) -> None:
        n_times = self.times.size
        require(self.marker_positions.shape[0] == n_times, "one marker frame per time")
        require(
            self.marker_positions.ndim == 3,
            "marker_positions must be (T, markers, dim)",
        )
        require(
            self.marker_sigma > 0.0, "marker_sigma must be positive", self.marker_sigma
        )
        require(self.basis.n_samples == n_times, "basis sample count must match times")
        require(
            bool(np.allclose(self.basis.times, self.times, rtol=0.0, atol=1e-12)),
            "basis grid must equal the trial times",
        )
        if self.phase_index is not None:
            require(self.phase_index.shape == (n_times,), "phase_index per time")

    @property
    def n_times(self) -> int:
        return int(self.times.size)


@dataclass(frozen=True)
class OuterPriors:
    """Priors and weights shared by all stages of the fit."""

    geometry_mean: FloatArray
    geometry_weight: FloatArray
    parameter_prior_mean: FloatArray
    parameter_prior_weight: FloatArray
    dynamics_sigma: FloatArray
    input_effort_weight: float
    input_smoothness_weight: float
    template_weight: float
    anchors: tuple[LinearAnchor, ...] = ()

    def __post_init__(self) -> None:
        require(
            self.geometry_mean.shape == self.geometry_weight.shape,
            "geometry prior shapes",
        )
        require(
            self.parameter_prior_mean.shape == self.parameter_prior_weight.shape,
            "pi prior shapes",
        )
        require(
            bool(np.all(self.dynamics_sigma > 0.0)), "dynamics_sigma must be positive"
        )
        for anchor in self.anchors:
            require(
                anchor.coefficients.shape == self.parameter_prior_mean.shape,
                "anchor width",
            )


@dataclass(frozen=True)
class OuterOptions:
    """Solver controls; ``continuation`` lists dynamics-weight factors in order."""

    continuation: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    max_iterations_per_stage: int = 30
    gradient_tolerance: float = 1e-6
    relative_cost_tolerance: float = 1e-10
    initial_damping: float = 1e-2
    finite_difference_step: float = 1e-6

    def __post_init__(self) -> None:
        require(len(self.continuation) >= 1, "continuation needs >= 1 factor")
        require(
            bool(np.all(np.diff(self.continuation) >= 0.0)),
            "continuation must be non-decreasing",
        )
        require(self.max_iterations_per_stage >= 1, "max_iterations_per_stage >= 1")
        require(self.finite_difference_step > 0.0, "finite_difference_step > 0")


@dataclass(frozen=True)
class StageReceipt:
    """Measured record of one continuation stage."""

    factor: float
    iterations: int
    cost_before: float
    cost_after: float
    wall_time_s: float
    converged: bool


@dataclass(frozen=True)
class OuterFit:
    """Result of :func:`fit_trials`."""

    coefficients: tuple[FloatArray, ...]
    geometry: FloatArray
    parameters: FloatArray
    inner: InnerSolution
    cost: float
    receipts: tuple[StageReceipt, ...]
    gradient_norm: float


@dataclass(frozen=True)
class _Context:
    factory: ModelFactory
    trials: tuple[TrialData, ...]
    priors: OuterPriors
    options: OuterOptions
    parameterization: InertialParameterization
    shapes: tuple[tuple[int, int], ...]
    n_geometry: int
    n_theta: int

    @property
    def offsets(self) -> list[int]:
        sizes = [rows * cols for rows, cols in self.shapes]
        return [0, *np.cumsum(sizes).tolist()]

    @property
    def geometry_columns(self) -> slice:
        return slice(self.offsets[-1], self.offsets[-1] + self.n_geometry)

    @property
    def theta_columns(self) -> slice:
        start = self.offsets[-1] + self.n_geometry
        return slice(start, start + self.n_theta)

    def unpack(self, xi: FloatArray) -> tuple[list[FloatArray], FloatArray, FloatArray]:
        offsets = self.offsets
        coefficients = [
            xi[offsets[k] : offsets[k + 1]].reshape(shape)
            for k, shape in enumerate(self.shapes)
        ]
        return coefficients, xi[self.geometry_columns], xi[self.theta_columns]

    def pack(
        self,
        coefficients: Sequence[FloatArray],
        geometry: FloatArray,
        theta: FloatArray,
    ) -> FloatArray:
        return np.concatenate([*(c.ravel() for c in coefficients), geometry, theta])


@dataclass(frozen=True)
class _Evaluation:
    xi: FloatArray
    model: RegressorModel
    geometry: FloatArray
    theta: FloatArray
    parameters: FloatArray
    coefficients: list[FloatArray]
    kinematics: list[tuple[FloatArray, FloatArray, FloatArray]]
    regressors: FloatArray
    inner: InnerSolution
    observation_residual: FloatArray
    outer_prior_residual: FloatArray

    @property
    def cost(self) -> float:
        return float(
            self.observation_residual @ self.observation_residual
            + self.inner.cost
            + self.outer_prior_residual @ self.outer_prior_residual
        )


def dynamics_jacobian_by_finite_differences(
    factory: ModelFactory,
    geometry: FloatArray,
    q: FloatArray,
    v: FloatArray,
    a: FloatArray,
    pi: FloatArray,
    step: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Central differences of ``tau = Y(q, v, a) pi`` w.r.t. ``q, v, a`` and geometry.

    Each perturbation is applied to *all* nodes at once, so the cost is
    ``2 (3 n_dof + n_geom)`` batched regressor evaluations.
    """
    require(step > 0.0, "step must be positive", step)
    model = factory(geometry)
    n_dof = q.shape[1]

    def torque(
        model_: RegressorModel, q_: FloatArray, v_: FloatArray, a_: FloatArray
    ) -> FloatArray:
        return model_.regressor(q_, v_, a_) @ pi

    def column(delta_index: int, which: int) -> FloatArray:
        bump = np.zeros_like(q)
        bump[:, delta_index] = step
        args = [q, v, a]
        plus, minus = list(args), list(args)
        plus[which] = args[which] + bump
        minus[which] = args[which] - bump
        return (torque(model, *plus) - torque(model, *minus)) / (2.0 * step)

    d_q, d_v, d_a = (
        np.stack([column(j, which) for j in range(n_dof)], axis=-1)
        for which in range(3)
    )
    d_geometry = np.stack(
        [
            (
                torque(factory(geometry + step * unit), q, v, a)
                - torque(factory(geometry - step * unit), q, v, a)
            )
            / (2.0 * step)
            for unit in np.eye(geometry.size)
        ],
        axis=-1,
    )
    return d_q, d_v, d_a, d_geometry


def _outer_prior_rows(
    ctx: _Context, geometry: FloatArray, parameters: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return ``(residual, d residual / d pi)`` of geometry prior, pi prior and anchors.

    The geometry rows carry a zero ``d/d pi`` block; their Jacobian w.r.t.
    geometry is the diagonal weight handled in :func:`_reduced_system`.
    """
    priors = ctx.priors
    anchor_rows = (
        np.stack([a.weight * a.coefficients for a in priors.anchors])
        if priors.anchors
        else np.zeros((0, parameters.size))
    )
    anchor_values = np.array([a.weight * a.value for a in priors.anchors])
    residual = np.concatenate(
        [
            priors.geometry_weight * (geometry - priors.geometry_mean),
            priors.parameter_prior_weight * (parameters - priors.parameter_prior_mean),
            anchor_rows @ parameters - anchor_values,
        ]
    )
    d_pi = np.concatenate(
        [
            np.zeros((geometry.size, parameters.size)),
            np.diag(priors.parameter_prior_weight),
            anchor_rows,
        ]
    )
    return residual, d_pi


def _stack_phases(phases: list[IntArray | None]) -> IntArray | None:
    present = [phase for phase in phases if phase is not None]
    if len(present) != len(phases):
        return None
    return np.concatenate(present)


def _evaluate(xi: FloatArray, ctx: _Context, factor: float) -> _Evaluation:
    coefficients, geometry, theta = ctx.unpack(xi)
    parameters = ctx.parameterization.from_unconstrained(theta)
    model = ctx.factory(geometry)
    kinematics = [
        trial.basis.evaluate(c)
        for trial, c in zip(ctx.trials, coefficients, strict=True)
    ]
    obs = [
        (
            (model.marker_positions(q, trial.markers) - trial.marker_positions)
            / trial.marker_sigma
        ).ravel()
        for trial, (q, _, _) in zip(ctx.trials, kinematics, strict=True)
    ]
    q_all, v_all, a_all = (np.concatenate([k[i] for k in kinematics]) for i in range(3))
    phases = [t.phase_index for t in ctx.trials]
    regressors = model.regressor(q_all, v_all, a_all)
    problem = InnerProblem(
        regressors=regressors,
        input_matrix=model.input_matrix,
        trial_index=np.repeat(
            np.arange(len(ctx.trials)), [t.n_times for t in ctx.trials]
        ),
        phase_index=_stack_phases(phases),
        dynamics_weight=factor / ctx.priors.dynamics_sigma,
        parameter_prior_mean=ctx.priors.parameter_prior_mean,
        parameter_prior_weight=ctx.priors.parameter_prior_weight,
        input_effort_weight=ctx.priors.input_effort_weight,
        input_smoothness_weight=ctx.priors.input_smoothness_weight,
        template_weight=ctx.priors.template_weight,
        fixed_parameters=parameters,
    )
    prior_residual, _ = _outer_prior_rows(ctx, geometry, parameters)
    return _Evaluation(
        xi,
        model,
        geometry,
        theta,
        parameters,
        coefficients,
        kinematics,
        regressors,
        solve_inner(problem),
        np.concatenate(obs),
        prior_residual,
    )


def _observation_jacobian(ev: _Evaluation, ctx: _Context) -> FloatArray:
    offsets = ctx.offsets
    blocks = []
    for k, (trial, (q, _, _)) in enumerate(zip(ctx.trials, ev.kinematics, strict=True)):
        jac_q, jac_geom = ev.model.marker_jacobians(q, trial.markers)
        rows = jac_q.shape[0] * jac_q.shape[1] * jac_q.shape[2]
        block = np.zeros((rows, ev.xi.size))
        block[:, offsets[k] : offsets[k + 1]] = np.einsum(
            "tmdj,tc->tmdcj", jac_q, trial.basis.position
        ).reshape(rows, -1)
        block[:, ctx.geometry_columns] = jac_geom.reshape(rows, -1)
        blocks.append(block / trial.marker_sigma)
    return np.concatenate(blocks)


def _dynamics_jacobian(ev: _Evaluation, ctx: _Context, factor: float) -> FloatArray:
    q_all, v_all, a_all = (
        np.concatenate([k[i] for k in ev.kinematics]) for i in range(3)
    )
    d_q, d_v, d_a, d_geom = dynamics_jacobian_by_finite_differences(
        ctx.factory,
        ev.geometry,
        q_all,
        v_all,
        a_all,
        ev.parameters,
        ctx.options.finite_difference_step,
    )
    weight = factor / ctx.priors.dynamics_sigma
    d_theta = ev.regressors @ ctx.parameterization.jacobian(ev.theta)
    offsets, start, blocks = ctx.offsets, 0, []
    for k, trial in enumerate(ctx.trials):
        stop = start + trial.n_times
        basis = trial.basis
        chained = (
            np.einsum("tij,tc->ticj", d_q[start:stop], basis.position)
            + np.einsum("tij,tc->ticj", d_v[start:stop], basis.velocity)
            + np.einsum("tij,tc->ticj", d_a[start:stop], basis.acceleration)
        )
        rows = trial.n_times * d_q.shape[1]
        block = np.zeros((rows, ev.xi.size))
        block[:, offsets[k] : offsets[k + 1]] = chained.reshape(rows, -1)
        block[:, ctx.geometry_columns] = d_geom[start:stop].reshape(rows, -1)
        block[:, ctx.theta_columns] = d_theta[start:stop].reshape(rows, -1)
        blocks.append(block * np.tile(weight, trial.n_times)[:, None])
        start = stop
    return np.concatenate(blocks)


def _outer_prior_jacobian(ev: _Evaluation, ctx: _Context) -> FloatArray:
    _, d_pi = _outer_prior_rows(ctx, ev.geometry, ev.parameters)
    jacobian = np.zeros((d_pi.shape[0], ev.xi.size))
    jacobian[: ev.geometry.size, ctx.geometry_columns] = np.diag(
        ctx.priors.geometry_weight
    )
    jacobian[:, ctx.theta_columns] = d_pi @ ctx.parameterization.jacobian(ev.theta)
    return jacobian


def _reduced_system(
    ev: _Evaluation, ctx: _Context, factor: float
) -> tuple[FloatArray, FloatArray]:
    jac_obs = _observation_jacobian(ev, ctx)
    jac_prior = _outer_prior_jacobian(ev, ctx)
    hessian = jac_obs.T @ jac_obs + jac_prior.T @ jac_prior
    gradient = (
        jac_obs.T @ ev.observation_residual + jac_prior.T @ ev.outer_prior_residual
    )
    if factor > 0.0:
        h_dyn, g_dyn = reduced_gauss_newton(
            ev.inner, _dynamics_jacobian(ev, ctx, factor)
        )
        hessian, gradient = hessian + h_dyn, gradient + g_dyn
    return hessian, gradient


def _try_step(
    ev: _Evaluation,
    hessian: FloatArray,
    gradient: FloatArray,
    damping: float,
    ctx: _Context,
    factor: float,
) -> tuple[_Evaluation | None, float, float]:
    """Return ``(accepted evaluation or None, gain ratio, predicted decrease)``."""
    scale = np.maximum(np.diag(hessian), _MIN_DIAGONAL)
    step = np.linalg.solve(hessian + damping * np.diag(scale), -gradient)
    candidate = _evaluate(ev.xi + step, ctx, factor)
    predicted = 0.5 * float(step @ (damping * scale * step - gradient))
    rho = (ev.cost - candidate.cost) / max(predicted, np.finfo(float).tiny)
    return (candidate if rho > 0.0 else None), rho, predicted


def _iterate(
    ev: _Evaluation, ctx: _Context, factor: float
) -> tuple[_Evaluation, int, bool, float]:
    """Run LM iterations for one stage; returns ``(ev, iterations, converged, |g|_inf)``."""
    options = ctx.options
    damping, growth, converged, gradient_norm, iterations = (
        options.initial_damping,
        2.0,
        False,
        np.inf,
        0,
    )
    for iterations in range(1, options.max_iterations_per_stage + 1):  # noqa: B007 - count reported
        hessian, gradient = _reduced_system(ev, ctx, factor)
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm < options.gradient_tolerance:
            converged = True
            break
        accepted, predicted = None, np.inf
        while accepted is None and damping < _DAMPING_CEILING:
            accepted, rho, predicted = _try_step(
                ev, hessian, gradient, damping, ctx, factor
            )
            if accepted is None:
                damping, growth = damping * growth, 2.0 * growth
            else:
                damping = max(
                    damping * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3),
                    _DAMPING_FLOOR,
                )
                growth = 2.0
        if accepted is None:
            converged = predicted <= options.relative_cost_tolerance * max(ev.cost, 1.0)
            break
        previous, ev = ev.cost, accepted
        if previous - ev.cost <= options.relative_cost_tolerance * max(previous, 1.0):
            converged = True
            break
    return ev, iterations, converged, gradient_norm


def _run_stage(
    xi: FloatArray, ctx: _Context, factor: float
) -> tuple[_Evaluation, StageReceipt, float]:
    started = time.perf_counter()
    initial = _evaluate(xi, ctx, factor)
    ev, iterations, converged, gradient_norm = _iterate(initial, ctx, factor)
    receipt = StageReceipt(
        factor,
        iterations,
        initial.cost,
        ev.cost,
        time.perf_counter() - started,
        converged,
    )
    logger.info(
        "mosaic stage factor=%.3g iters=%d cost %.4g -> %.4g",
        factor,
        iterations,
        initial.cost,
        ev.cost,
    )
    return ev, receipt, gradient_norm


def fit_trials(
    factory: ModelFactory,
    trials: Sequence[TrialData],
    initial_coefficients: Sequence[FloatArray],
    initial_geometry: FloatArray,
    initial_parameters: FloatArray,
    parameterization: InertialParameterization,
    priors: OuterPriors,
    options: OuterOptions,
) -> OuterFit:
    """Jointly fit kinematics, geometry, inertial parameters and inputs.

    Preconditions: one coefficient matrix per trial with ``basis.n_coefficients``
    rows and ``n_dof`` columns; geometry and parameters match the priors and the
    initial parameters are strictly physically consistent.  Postcondition: the
    stage costs are non-increasing within each stage and the returned
    parameters are physically consistent by construction.
    """
    require(len(trials) >= 1, "at least one trial")
    require(
        len(initial_coefficients) == len(trials), "one coefficient matrix per trial"
    )
    model = factory(initial_geometry)
    for trial, coefficients in zip(trials, initial_coefficients, strict=True):
        require(
            coefficients.shape == (trial.basis.n_coefficients, model.n_dof),
            "coefficient shape",
            coefficients.shape,
        )
    require(
        initial_geometry.shape == priors.geometry_mean.shape, "geometry/prior shape"
    )
    require(
        initial_parameters.shape == priors.parameter_prior_mean.shape,
        "parameters/prior shape",
    )
    theta0 = parameterization.to_unconstrained(initial_parameters)
    ctx = _Context(
        factory,
        tuple(trials),
        priors,
        options,
        parameterization,
        tuple(c.shape for c in initial_coefficients),
        initial_geometry.size,
        theta0.size,
    )
    xi = ctx.pack(initial_coefficients, initial_geometry, theta0)
    receipts: list[StageReceipt] = []
    gradient_norm = np.inf
    for factor in options.continuation:
        ev, receipt, gradient_norm = _run_stage(xi, ctx, factor)
        receipts.append(receipt)
        xi = ev.xi
    ensure(
        all(r.cost_after <= r.cost_before + 1e-12 for r in receipts),
        "stage costs non-increasing",
    )
    final_coefficients, geometry, theta = ctx.unpack(xi)
    return OuterFit(
        tuple(final_coefficients),
        geometry,
        parameterization.from_unconstrained(theta),
        ev.inner,
        ev.cost,
        tuple(receipts),
        gradient_norm,
    )
