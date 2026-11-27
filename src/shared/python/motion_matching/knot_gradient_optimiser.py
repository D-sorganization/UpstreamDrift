"""JAX-free knot basis, horizon mask, and Adam optimiser driver.

This module provides the core array algorithms for gradient-based knot optimisation
of motion trajectories without requiring JAX. It can operate on either NumPy arrays
or JAX arrays via the array namespace protocol.

Exported symbols:
    - knot_grid: Generate a 1-D uniform knot grid.
    - knot_basis: Linear hat-function basis matrix.
    - horizon_knot_mask: Active-knot mask for cost horizon truncation.
    - AdamSettings: Hyperparameter configuration for Adam.
    - LbfgsSettings: Hyperparameter configuration for L-BFGS-B.
    - KnotOptimisationResult: Optimisation result container.
    - adam_minimise: Bias-corrected Adam minimiser loop.
    - lbfgs_minimise: SciPy L-BFGS-B minimiser wrapper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np

from src.shared.python.core.contracts import require


def knot_grid(times: np.ndarray, spacing_s: float) -> np.ndarray:
    """Generate a 1-D uniform knot grid covering the given time span.

    The knot count is computed as ``ceil((t_end - t0) / spacing_s - 1e-9) + 1``
    to prevent untouched trailing knots when the span is an exact multiple.

    Preconditions:
        - times is 1-D, finite, strictly increasing with at least 2 samples.
        - spacing_s is positive and finite.
    """
    t = np.asarray(times)
    require(bool(t.ndim == 1), "times must be 1-D", t.ndim)
    require(bool(t.size >= 2), "times must have at least 2 samples", t.size)
    require(bool(np.all(np.isfinite(t))), "times must be finite")
    require(bool(np.all(np.diff(t) > 0.0)), "times must be strictly increasing")
    require(
        bool(np.isfinite(spacing_s) and spacing_s > 0.0),
        "spacing_s must be positive and finite",
        spacing_s,
    )

    t0 = float(t[0])
    t_end = float(t[-1])
    n = int(np.ceil((t_end - t0) / spacing_s - 1e-9)) + 1
    return t0 + spacing_s * np.arange(n, dtype=float)


def knot_basis(times: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Evaluate the linear hat-function basis matrix (frames x knots).

    Every row sums to 1.0, and the basis is exact at knot times. Refuses
    untouched knots where any column sum is 0.

    Preconditions:
        - times is 1-D, finite, strictly increasing with at least 1 sample.
        - knots is 1-D, finite, strictly increasing with at least 2 samples.
        - times falls within [knots[0], knots[-1]].
        - Every knot is touched by at least one frame (column sum > 0).
    """
    t = np.asarray(times)
    k = np.asarray(knots)
    require(bool(t.ndim == 1), "times must be 1-D", t.ndim)
    require(bool(t.size >= 1), "times must be non-empty", t.size)
    require(bool(np.all(np.isfinite(t))), "times must be finite")
    if t.size >= 2:
        require(bool(np.all(np.diff(t) > 0.0)), "times must be strictly increasing")

    require(bool(k.ndim == 1), "knots must be 1-D", k.ndim)
    require(bool(k.size >= 2), "knots must have at least 2 samples", k.size)
    require(bool(np.all(np.isfinite(k))), "knots must be finite")
    require(bool(np.all(np.diff(k) > 0.0)), "knots must be strictly increasing")

    require(bool(t[0] >= k[0]), "times[0] must not precede knots[0]")
    require(bool(t[-1] <= k[-1]), "times[-1] must not exceed knots[-1]")

    n_frames = t.size
    n_knots = k.size

    # Find interval for each time, clamping the last frame into the final interval
    j = np.clip(np.searchsorted(k, t, side="right") - 1, 0, n_knots - 2)
    dt = k[j + 1] - k[j]
    w = (t - k[j]) / dt

    basis = np.zeros((n_frames, n_knots), dtype=float)
    row_indices = np.arange(n_frames)
    basis[row_indices, j] = 1.0 - w
    basis[row_indices, j + 1] = w

    col_sums = np.sum(basis, axis=0)
    require(
        bool(np.all(col_sums > 0.0)),
        "Every knot must be touched by at least one frame (column sum > 0)",
    )
    return basis


def horizon_knot_mask(knots: np.ndarray, horizon_s: float) -> np.ndarray:
    """Generate a binary float mask for knots within the time horizon.

    Knots at or before ``horizon_s`` are 1.0; knots after ``horizon_s`` are 0.0.

    Preconditions:
        - knots is 1-D and finite.
        - horizon_s is positive and finite.
    """
    require(
        bool(np.isfinite(horizon_s) and horizon_s > 0.0),
        "horizon_s must be positive and finite",
        horizon_s,
    )
    k = np.asarray(knots)
    require(bool(k.ndim == 1), "knots must be 1-D", k.ndim)
    require(bool(np.all(np.isfinite(k))), "knots must be finite")
    return (k <= horizon_s).astype(float)


@dataclass(frozen=True, kw_only=True)
class AdamSettings:
    """Configuration settings for the Adam optimiser (keyword-only)."""

    learning_rate: float
    max_iterations: int
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        require(
            self.learning_rate > 0.0,
            "learning_rate must be positive",
            self.learning_rate,
        )
        require(0.0 <= self.beta1 < 1.0, "beta1 must be in [0, 1)", self.beta1)
        require(0.0 <= self.beta2 < 1.0, "beta2 must be in [0, 1)", self.beta2)
        require(self.eps > 0.0, "eps must be positive", self.eps)
        require(
            self.max_iterations >= 0,
            "max_iterations must be non-negative",
            self.max_iterations,
        )


@dataclass(frozen=True, kw_only=True)
class LbfgsSettings:
    """Configuration settings for the L-BFGS-B optimiser (keyword-only).

    ``parameter_scale`` multiplies the internal iterate so SciPy optimizes ``y``
    with ``x = y * parameter_scale`` (physical ``delta_knots`` are ``x``). This
    puts typical rad-scale knot corrections on an O(1) scale so relative ``ftol``
    / ``gtol`` tests behave as intended (#13779). ``1.0`` disables scaling.
    """

    max_iterations: int
    max_evaluations: int | None = None
    parameter_scale: float = 1.0
    ftol: float | None = None
    gtol: float | None = None
    pgtol: float | None = None

    def __post_init__(self) -> None:
        require(
            self.max_iterations > 0,
            "max_iterations must be positive",
            self.max_iterations,
        )
        if self.max_evaluations is not None:
            require(
                self.max_evaluations > 0,
                "max_evaluations must be positive",
                self.max_evaluations,
            )
        require(
            bool(np.isfinite(self.parameter_scale) and self.parameter_scale > 0.0),
            "parameter_scale must be finite and positive",
            self.parameter_scale,
        )
        for name in ("ftol", "gtol", "pgtol"):
            v = getattr(self, name)
            if v is not None:
                require(
                    bool(np.isfinite(v) and v > 0.0),
                    f"{name} must be finite and positive when set",
                    v,
                )


StopReason = Literal[
    "max_iterations",
    "non_finite_cost",
    "non_finite_gradient",
    "converged",
    "max_evaluations",
    "error",
]

# A non-finite trial is shown to SciPy as this multiple of the worst finite total
# seen so far. Infinity with a zero gradient makes L-BFGS-B declare convergence on
# the spot, ending the run; a finite penalty makes the line search back off.
_NON_FINITE_PENALTY_FACTOR = 1.0e3


@dataclass(frozen=True)
class KnotOptimisationResult:
    """Result of knot gradient optimisation (Adam or L-BFGS-B).

    L-BFGS-B also fills ``evaluation_diagnostics`` (one row per history row:
    iteration, ``objective_s``, ``gradient_s``, ``projected_gradient_inf_norm``)
    and ``projected_gradient_norm`` (infinity norm at SciPy's final point, the
    quantity its ``pgtol`` test compares). Adam leaves them empty and ``None``.
    """

    best_x: Any
    best_objective: float
    best_iteration: int
    history: tuple[dict[str, float], ...]
    stop_reason: StopReason
    non_finite_encountered: bool = False
    evaluation_diagnostics: tuple[dict[str, float], ...] = ()
    projected_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.best_x, np.ndarray):
            self.best_x.setflags(write=False)


ValueAndGrad = Callable[[Any], tuple[tuple[Any, Any], Any]]


def _cost_row(
    total: Any, objective: Any, x: Any, k: int, xp: Any
) -> tuple[dict[str, float], bool]:
    """The history row of one evaluation and whether its cost is finite."""
    row = {
        "iteration": k,
        "total": float(total),
        "objective": float(objective),
        "max_abs_x": float(xp.max(xp.abs(x))),
    }
    return row, bool(np.isfinite(row["total"]) and np.isfinite(row["objective"]))


def _evaluate(
    value_and_grad: ValueAndGrad, x: Any, k: int, xp: Any
) -> tuple[dict[str, float], Any, bool, bool]:
    """Evaluate once; return (history row, grad, cost finite, grad finite)."""
    (total, objective), grad = value_and_grad(x)
    row, cost_ok = _cost_row(total, objective, x, k, xp)
    return row, grad, cost_ok, bool(xp.all(xp.isfinite(grad)))


def _timed_evaluation(
    value_and_grad: ValueAndGrad, x: np.ndarray, k: int, clock: Callable[[], float]
) -> tuple[dict[str, float], np.ndarray, bool, dict[str, float]]:
    """Evaluate once for L-BFGS-B; return (row, host grad, cost finite, timing).

    ``objective_s`` is the ``clock`` time until the cost scalars are on the host
    (the call, with any JIT compile or async compute it waits on). The gradient
    is not touched before that, so its transfer to a float64 host array is
    ``gradient_s``, never ``objective_s``. ``projected_gradient_inf_norm`` is
    ``max|g|``; with no bounds the projection is the identity.
    """
    start = clock()
    (total, objective), grad = value_and_grad(x)
    row, cost_ok = _cost_row(total, objective, x, k, np)
    costed = clock()
    grad_flat = np.asarray(grad, dtype=np.float64).ravel()
    timing = {
        "iteration": k,
        "objective_s": costed - start,
        "gradient_s": clock() - costed,
        "projected_gradient_inf_norm": float(np.max(np.abs(grad_flat))),
    }
    return row, grad_flat, cost_ok, timing


def adam_minimise(
    value_and_grad: ValueAndGrad,
    x0: Any,
    settings: AdamSettings,
    *,
    xp: Any = np,
    on_iteration: Callable[[int, Any, float, float], None] | None = None,
) -> KnotOptimisationResult:
    """Minimise an objective with bias-corrected Adam, keeping the best iterate.

    ``value_and_grad(x)`` returns ``((total, objective), grad)``: Adam descends
    ``total`` through ``grad`` while the best iterate is chosen by ``objective``
    (the prototype's marker cost). Row 0 of the history is the evaluation at
    ``x0``. A non-finite cost stops without accepting that iterate; a non-finite
    gradient stops after accepting it if its objective is finite and lower.
    ``xp`` is the array namespace (numpy or jax.numpy); only ``zeros_like``,
    ``sqrt``, ``isfinite``, ``all``, ``abs`` and ``max`` are called on it.
    """
    require(callable(value_and_grad), "value_and_grad must be callable")
    require(isinstance(settings, AdamSettings), "settings must be AdamSettings")
    require(bool(xp.all(xp.isfinite(x0))), "x0 must be finite")
    if isinstance(x0, np.ndarray):
        x0 = x0.copy()  # never freeze or alias the caller's array

    lr, b1, b2, eps = (
        settings.learning_rate,
        settings.beta1,
        settings.beta2,
        settings.eps,
    )
    row, grad, cost_ok, grad_ok = _evaluate(value_and_grad, x0, 0, xp)
    history = [row]
    if on_iteration is not None:
        on_iteration(0, x0, row["total"], row["objective"])
    best_x = x0
    best_objective = row["objective"] if cost_ok else float("inf")
    best_iteration = 0
    stop: StopReason = "max_iterations"
    if not cost_ok:
        stop = "non_finite_cost"
    elif not grad_ok:
        stop = "non_finite_gradient"

    m = xp.zeros_like(x0)
    v = xp.zeros_like(x0)
    x = x0
    k = 0
    while stop == "max_iterations" and k < settings.max_iterations:
        k += 1
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * grad**2
        x = x - lr * (m / (1.0 - b1**k)) / (xp.sqrt(v / (1.0 - b2**k)) + eps)
        row, grad, cost_ok, grad_ok = _evaluate(value_and_grad, x, k, xp)
        history.append(row)
        if on_iteration is not None:
            on_iteration(k, x, row["total"], row["objective"])
        if not cost_ok:
            stop = "non_finite_cost"
            break
        if row["objective"] < best_objective:
            best_x, best_objective, best_iteration = x, row["objective"], k
        if not grad_ok:
            stop = "non_finite_gradient"

    return KnotOptimisationResult(
        best_x=best_x,
        best_objective=best_objective,
        best_iteration=best_iteration,
        history=tuple(history),
        stop_reason=stop,
    )


def _lbfgs_scipy_options(settings: LbfgsSettings) -> dict[str, Any]:
    """SciPy ``minimize`` options for L-BFGS-B (typed loosely for stub compatibility)."""
    max_fun = 15000 if settings.max_evaluations is None else settings.max_evaluations
    options: dict[str, Any] = {
        "maxiter": settings.max_iterations,
        "maxfun": max_fun,
    }
    if settings.ftol is not None:
        options["ftol"] = settings.ftol
    if settings.gtol is not None:
        options["gtol"] = settings.gtol
    if settings.pgtol is not None:
        options["pgtol"] = settings.pgtol
    return options


def _scipy_stop_reason(
    res: Any, n_evaluations: int, settings: LbfgsSettings, non_finite: bool
) -> StopReason:
    """Map a SciPy L-BFGS-B result onto ``StopReason``, failing closed.

    Only an explicit iteration-limit message yields ``max_iterations``; any exit
    SciPy does not describe as success or as a budget stop is ``error``.
    """
    message = str(res.message).upper()
    if non_finite:
        return "non_finite_cost"
    if res.success:
        return "converged"
    if "EVALUATIONS" in message or (
        settings.max_evaluations is not None
        and n_evaluations >= settings.max_evaluations
    ):
        return "max_evaluations"
    if "ITERATIONS" in message:
        return "max_iterations"
    return "error"


@dataclass
class _LbfgsWorkspace:
    """Mutable scratch for one :func:`lbfgs_minimise` run (closure state)."""

    history: list[dict[str, float]]
    diagnostics: list[dict[str, float]]
    best_x: Any
    best_objective: float
    best_iteration: int
    non_finite_encountered: bool
    worst_finite_total: float | None


def _lbfgs_scaled_fun(
    value_and_grad: ValueAndGrad,
    shape: tuple[int, ...],
    scale: float,
    clock: Callable[[], float],
    on_iteration: Callable[[int, Any, float, float], None] | None,
    ws: _LbfgsWorkspace,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    """Build ``fun(y)`` with ``x = scale * y``; updates ``ws`` and ``ws.history``."""

    def fun(y_flat: np.ndarray) -> tuple[float, np.ndarray]:
        k = len(ws.history)
        x_flat = np.asarray(y_flat, dtype=np.float64) * scale
        x = x_flat.reshape(shape)
        row, grad_flat, cost_ok, timing = _timed_evaluation(value_and_grad, x, k, clock)
        grad_y = grad_flat * scale
        ws.diagnostics.append(timing)
        ws.history.append(row)
        if on_iteration is not None:
            on_iteration(k, x, row["total"], row["objective"])
        if not cost_ok:
            ws.non_finite_encountered = True
            worst = ws.worst_finite_total
            penalty = (
                float("inf")
                if worst is None
                else _NON_FINITE_PENALTY_FACTOR * (1.0 + worst)
            )
            return penalty, np.zeros_like(y_flat, dtype=np.float64)
        ws.worst_finite_total = max(ws.worst_finite_total or 0.0, abs(row["total"]))
        if row["objective"] < ws.best_objective:
            ws.best_x = x.copy()
            ws.best_objective = row["objective"]
            ws.best_iteration = k
        return float(row["total"]), grad_y

    return fun


def lbfgs_minimise(
    value_and_grad: ValueAndGrad,
    x0: Any,
    settings: LbfgsSettings,
    *,
    on_iteration: Callable[[int, Any, float, float], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> KnotOptimisationResult:
    """SciPy L-BFGS-B on ``value_and_grad``; optional ``parameter_scale`` reparam (#13779).

    Contract matches the historical wrapper: ``history[0]`` is ``x0``, non-finite
    trials become finite penalties, stop reasons are fail-closed, diagnostics
    use :func:`_timed_evaluation`. See issue #13779 for scaling and tolerances.
    """
    require(callable(value_and_grad), "value_and_grad must be callable")
    require(isinstance(settings, LbfgsSettings), "settings must be LbfgsSettings")
    require(bool(np.all(np.isfinite(x0))), "x0 must be finite")
    if on_iteration is not None:
        require(callable(on_iteration), "on_iteration must be callable")
    require(callable(clock), "clock must be callable")

    import scipy.optimize

    scale = float(settings.parameter_scale)
    shape = np.shape(x0)
    x0_arr = np.asarray(x0, dtype=np.float64)
    y0_flat = (x0_arr.ravel().copy() / scale).copy()

    ws = _LbfgsWorkspace(
        history=[],
        diagnostics=[],
        best_x=x0_arr.copy(),
        best_objective=float("inf"),
        best_iteration=0,
        non_finite_encountered=False,
        worst_finite_total=None,
    )

    fun = _lbfgs_scaled_fun(
        value_and_grad,
        shape,
        scale,
        clock,
        on_iteration,
        ws,
    )

    opts: Any = _lbfgs_scipy_options(settings)
    res = scipy.optimize.minimize(
        fun,
        y0_flat,
        jac=True,
        method="L-BFGS-B",
        options=opts,
    )

    stop = _scipy_stop_reason(res, len(ws.history), settings, ws.non_finite_encountered)
    jac_y = np.asarray(res.jac, dtype=np.float64)
    projected_x_space = float(np.max(np.abs(jac_y))) / scale

    return KnotOptimisationResult(
        cast(Any, ws.best_x),
        ws.best_objective,
        ws.best_iteration,
        tuple(ws.history),
        stop,
        non_finite_encountered=ws.non_finite_encountered,
        evaluation_diagnostics=tuple(ws.diagnostics),
        projected_gradient_norm=projected_x_space,
    )
