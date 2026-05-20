"""Drake gradient-free fit driver (cross-engine §2.4 / issue #4115).

This module exposes :func:`fit_swing_drake` — the default Drake
swing-fit driver. It wraps :func:`scipy.optimize.minimize` with
``method="SLSQP"`` (a bounds-aware quasi-Newton method) over the
canonical 7-coefficient-per-joint torque polynomial, with the inner cost
evaluation routed through
:func:`src.shared.python.motion_matching.cost.compute_cost` via the
:mod:`compute_cost_drake` adapter.

Per the canonical contract (cross-engine §2.4 / DRAKE_PARITY_SPEC §2.3)
the polynomial bounds are:

* ``|A_j|`` and ``|B_j|`` ≤ 1000
* ``|C_j|`` and ``|D_j|`` ≤ 500
* ``|E_j|`` and ``|F_j|`` ≤ 100
* ``|G_j|`` ≤ 25

These mirror ``getPolynomialParameterInfo`` in the Simscape reference
and fall in the same boxed search space the fmincon / surrogate / NN
fitters use.

The default ``simulate_fn`` is a closure over
:func:`simulate_with_coefficients` (the float Drake forward sim from
issue #4111). Tests can inject a stub ``simulate_fn`` to exercise the
optimizer wiring without a live Drake plant.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.cost import CostOptions
from src.shared.python.motion_matching.fit_result import CanonicalFitResult as FitResult
from src.shared.python.motion_matching.validate_theta import validate_theta

from .compute_cost_drake import compute_cost_drake
from .simulate import COEFFS_PER_JOINT, SimOptions, SimOut, simulate_with_coefficients

logger = logging.getLogger(__name__)

__all__ = [
    "FitOptions",
    "FitResult",
    "fit_swing_drake",
    "polynomial_parameter_bounds",
]


# ---------------------------------------------------------------------------
# Canonical polynomial bounds (cross-engine §2.4)
# ---------------------------------------------------------------------------

#: Per-coefficient absolute bounds, indexed A..G in the canonical packing
#: order ``[A, B, C, D, E, F, G]``.
_PER_COEFF_ABS_BOUND: tuple[float, ...] = (
    1000.0,  # A
    1000.0,  # B
    500.0,  # C
    500.0,  # D
    100.0,  # E
    100.0,  # F
    25.0,  # G
)


def polynomial_parameter_bounds(
    n_joints: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the canonical ``(lower, upper)`` bound vectors for ``theta``.

    Args:
        n_joints: Number of actuated joints. ``theta`` has length
            ``n_joints * 7`` in canonical packing order.

    Returns:
        ``(lower, upper)`` arrays of shape ``(n_joints * 7,)``. Element
        ``k`` of each array bounds the corresponding ``theta[k]``.

    Raises:
        ValueError: If ``n_joints`` is non-positive.
    """
    if n_joints <= 0:
        msg = f"n_joints must be a positive integer; got {n_joints}"
        raise ValueError(msg)
    block = np.asarray(_PER_COEFF_ABS_BOUND, dtype=np.float64)
    upper = np.tile(block, n_joints)
    lower = -upper
    return lower, upper


# ---------------------------------------------------------------------------
# FitOptions / FitResult dataclasses (cross-engine canonical schema)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitOptions:
    """Options for :func:`fit_swing_drake`.

    Attributes:
        n_joints: Number of actuated joints. ``theta`` has length
            ``n_joints * 7``. Default 23 — the canonical Drake humanoid.
        method: scipy method name. Default ``"SLSQP"`` (bounds-aware
            quasi-Newton).
        max_iterations: scipy ``maxiter`` cap. Default 200; pass ~50 for
            tests where the per-eval cost is dominated by a real Drake
            sim (the spec budget is < 5 minutes per swing).
        tolerance: scipy ``ftol``. Default 1e-6.
        theta0: Optional warm-start vector of length ``n_joints * 7``.
            ``None`` selects the canonical zero-vector start.
        rng_seed: Reserved for stochastic warm-starts (currently unused;
            included so the schema is forward-compatible with the
            multistart driver in issue #4117).
        sim_options: Options forwarded to :func:`simulate_with_coefficients`.
        cost_options: :class:`CostOptions` forwarded to the shared cost.
        verbose: Log per-evaluation cost when True.
    """

    n_joints: int = 23
    method: str = "SLSQP"
    max_iterations: int = 200
    tolerance: float = 1.0e-6
    theta0: NDArray[np.float64] | None = None
    rng_seed: int = 0
    sim_options: SimOptions = field(default_factory=SimOptions)
    cost_options: CostOptions = field(default_factory=CostOptions)
    verbose: bool = False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


SimulateFn = Callable[[NDArray[np.float64]], SimOut]


def _default_simulate_fn(sim_options: SimOptions) -> SimulateFn:
    """Closure over :func:`simulate_with_coefficients` for SLSQP's hot loop."""

    def _run(theta: NDArray[np.float64]) -> SimOut:
        return simulate_with_coefficients(theta, sim_options)

    return _run


def _scipy_status(res: Any) -> str:
    """Map a ``scipy.OptimizeResult`` to the canonical solver_status string."""
    if bool(res.success):
        return "success"
    # SLSQP returns success=False when maxiter is hit but the iterate is
    # often still usable; classify those as warning rather than failed.
    msg = (str(res.message) or "").lower()
    if "iteration" in msg or "maximum" in msg:
        return "warning"
    return "failed"


def _final_rmse_m(
    theta: NDArray[np.float64], target: ClubTarget, sim_fn: SimulateFn
) -> float:
    """Recompute the unweighted position-RMSE in metres for reporting.

    Mirrors the ``_position_term`` in the shared cost but takes its
    square root and is unweighted, so callers can compare directly to
    the ``< 5 mm`` acceptance gate from cross-engine §2.4.
    """
    sim_out = sim_fn(theta)
    db = np.asarray(sim_out.grip) - np.asarray(target.butt)
    dc = np.asarray(sim_out.clubhead) - np.asarray(target.clubhead)
    if db.shape[0] == 0:
        return float("nan")
    # ⚡ Bolt: np.vdot is ~3-4x faster than np.sum(x*x, axis=1)
    # and avoids temporary allocations
    return float(np.sqrt((np.vdot(db, db) + np.vdot(dc, dc)) / db.shape[0]))


def fit_swing_drake(
    target: ClubTarget,
    options: FitOptions | None = None,
    *,
    simulate_fn: SimulateFn | None = None,
) -> FitResult:
    """Default Drake fit driver: ``scipy.optimize.minimize(method=SLSQP)``.

    Routes the inner cost evaluation through
    :func:`src.shared.python.motion_matching.cost.compute_cost` via the
    Drake adapter; ``simulate_fn`` defaults to a closure over
    :func:`simulate_with_coefficients` (the float Drake forward sim).

    Args:
        target: Validated :class:`ClubTarget` from the loaders module.
        options: :class:`FitOptions`; defaults to
            ``FitOptions(n_joints=23)`` with SLSQP, max 200 iters.
        simulate_fn: Optional override for the forward sim. The default
            calls real pydrake; tests inject a deterministic stub to
            exercise the optimizer wiring.

    Returns:
        :class:`FitResult` with ``theta_optimal``, ``final_rmse_m``,
        ``solver_status``, ``iterations``, ``wall_clock_s``, etc.

    Raises:
        ValueError: If ``options.theta0`` (when supplied) does not have
            the expected length, or if ``options.n_joints`` is non-positive.
    """
    from scipy.optimize import minimize  # noqa: PLC0415 - heavy lazy import

    opts = options if options is not None else FitOptions()
    if opts.n_joints <= 0:
        raise ValueError(f"n_joints must be positive; got {opts.n_joints}")

    n_dim = opts.n_joints * COEFFS_PER_JOINT
    lb, ub = polynomial_parameter_bounds(opts.n_joints)

    if opts.theta0 is not None:
        theta0 = np.ascontiguousarray(opts.theta0, dtype=np.float64).reshape(-1)
        if theta0.size != n_dim:
            msg = (
                f"theta0 has length {theta0.size}, expected {n_dim} "
                f"(= n_joints * {COEFFS_PER_JOINT})"
            )
            raise ValueError(msg)
        # Clip to the bound box defensively so SLSQP starts feasible.
        theta0 = np.clip(theta0, lb, ub).astype(np.float64)
    else:
        theta0 = np.zeros(n_dim, dtype=np.float64)

    sim_fn: SimulateFn = (
        simulate_fn
        if simulate_fn is not None
        else (_default_simulate_fn(opts.sim_options))
    )

    history: list[float] = []
    n_eval = 0

    def _objective(theta: NDArray[np.float64]) -> float:
        nonlocal n_eval
        n_eval += 1
        j, _terms = compute_cost_drake(theta, target, sim_fn, opts.cost_options)
        history.append(float(j))
        if opts.verbose:
            logger.info("fit_swing_drake n_eval=%d J=%.6e", n_eval, j)
        return float(j)

    bounds = list(zip(lb.tolist(), ub.tolist(), strict=False))

    t_start = time.perf_counter()
    res = minimize(
        _objective,
        theta0,
        method=opts.method,
        bounds=bounds,
        options={"maxiter": opts.max_iterations, "ftol": opts.tolerance},
    )
    wall_clock_s = time.perf_counter() - t_start

    # Spec §2.2: validate the recovered ``theta_optimal`` before we
    # advertise it as a fit. Out-of-bounds is the bigger risk for SLSQP
    # (it can step outside the feasible region on a hard failure), so we
    # also enforce the per-letter bounds the optimizer was told to use.
    bound_table = {
        chr(ord("A") + col): (-_PER_COEFF_ABS_BOUND[col], _PER_COEFF_ABS_BOUND[col])
        for col in range(COEFFS_PER_JOINT)
    }
    theta_opt = validate_theta(
        np.ascontiguousarray(res.x, dtype=np.float64),
        n_joints=opts.n_joints,
        bounds=bound_table,
        name="theta_optimal",
    )
    try:
        rmse_m = _final_rmse_m(theta_opt, target, sim_fn)
    except Exception:  # pragma: no cover - defensive  # noqa: BLE001
        rmse_m = float("nan")

    return FitResult(
        theta_optimal=theta_opt,
        final_cost=float(res.fun),
        final_rmse_m=rmse_m,
        solver_status=_scipy_status(res),
        iterations=int(getattr(res, "nit", 0)),
        n_evaluations=n_eval,
        wall_clock_s=wall_clock_s,
        message=str(res.message),
        history=tuple(history),
        method=opts.method,
        git_commit="unknown",
        engine_version="unknown",
        target_hash="unknown",
        timestamp_utc="unknown",
    )
