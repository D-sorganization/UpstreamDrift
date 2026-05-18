"""``fit_swing_opensim`` motion-matching driver (issue #4128).

Canonical OpenSim fit driver that recovers polynomial torque coefficients
from a measured club trajectory by minimising the engine-agnostic
``compute_cost`` (``shared/python/motion_matching/cost.py``) against a
forward simulator.

Optimizer
---------
``scipy.optimize.minimize(method="SLSQP")`` with explicit polynomial
coefficient bounds. Rationale:

* SLSQP supports the box constraints SLSQP/L-BFGS-B both expose, but unlike
  L-BFGS-B it can be extended with arbitrary equality / inequality
  constraints later without changing the driver shape.
* Finite-difference gradients are acceptable because the forward
  simulation dominates the per-iteration cost (tens of seconds vs.
  microseconds for the gradient finite-differencing).
* Performance target (per OPENSIM_PARITY_SPEC § fit driver): equal or
  faster than the Simscape baseline of ~7 s warm-sim × ~30 SLSQP
  iterations ≈ 4 minutes wall-clock for the 1.0 s recovery test.

Cost function
-------------
This driver imports ``compute_cost`` from
``shared/python/motion_matching/cost.py`` — **never** writes its own cost
(per cross-engine spec § 2.3). The forward simulator is supplied either by
the caller via ``options.simulate_fn`` (the TDD seam) or, when omitted,
via the OpenSim ``simulate_with_coefficients`` wrapper from issue #4120.

Public API
----------
    fit_swing_opensim(target, options=None) -> FitResult
    FitOptions
    FitResult
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
from src.shared.python.motion_matching.cost import (
    CostOptions,
    SimOutput,
    compute_cost,
)
from src.shared.python.motion_matching.fit_result import CanonicalFitResult as FitResult
from src.shared.python.motion_matching.validate_theta import validate_theta

logger = logging.getLogger(__name__)

# Number of polynomial coefficients per joint (matches Simscape reference and
# OPENSIM_PARITY_SPEC § 2.2: ``tau_j(t) = sum_{k=0..6} theta[7*j+k] * t**k``).
POLY_ORDER_PER_JOINT: int = 7

# Default per-coefficient bounds. Mirrors the Simscape adapter polynomial
# bounds (broad enough to span realistic peak torques, tight enough to keep
# the SLSQP search well-conditioned). See issue #4128 acceptance criteria.
DEFAULT_COEFF_BOUND: float = 50.0

# Default number of joints when the simulate function does not advertise an
# ``n_joints`` attribute. Matches the canonical 25-DOF golf humanoid.
DEFAULT_N_JOINTS: int = 25


__all__ = [
    "DEFAULT_COEFF_BOUND",
    "DEFAULT_N_JOINTS",
    "POLY_ORDER_PER_JOINT",
    "FitOptions",
    "FitResult",
    "fit_swing_opensim",
]


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


@dataclass
class FitOptions:
    """Options accepted by :func:`fit_swing_opensim`.

    Attributes:
        method:        scipy optimizer method. Default ``"SLSQP"`` per spec.
        max_iter:      Optimizer iteration cap. Default ``200``; tests should
                       set ~30 to keep the recovery test under the target.
        ftol:          scipy ``ftol`` for the objective. Default ``1e-6``.
        coeff_bound:   Symmetric per-coefficient bound. ``theta_k`` is
                       constrained to ``[-coeff_bound, +coeff_bound]``.
        n_joints:      Number of joints (sets the dimensionality
                       ``d = n_joints * 7``). When ``None``, inferred from
                       ``simulate_fn.n_joints`` if present, else
                       :data:`DEFAULT_N_JOINTS`.
        theta0:        Optional warm-start vector of length ``d``.
        rng_seed:      RNG seed for the warm-start draw when ``theta0`` is
                       ``None``. Determinism is part of the contract.
        simulate_fn:   Forward simulator. Signature:
                       ``simulate_fn(theta) -> SimOutput``. Defaults to the
                       OpenSim ``simulate_with_coefficients`` wrapper (issue
                       #4120). Tests inject a deterministic mock.
        cost_options:  :class:`CostOptions` overrides (defaults are spec
                       defaults).
        verbose:       Log per-iteration cost when True.
    """

    method: str = "SLSQP"
    max_iter: int = 200
    ftol: float = 1e-6
    coeff_bound: float = DEFAULT_COEFF_BOUND
    n_joints: int | None = None
    theta0: NDArray[np.float64] | None = None
    rng_seed: int = 42
    simulate_fn: Callable[[NDArray[np.float64]], SimOutput] | None = None
    cost_options: CostOptions = field(
        # ``coeff_l2`` is the only regularizer that does not require the
        # simulator to emit ``tau``/``omega``. The OpenSim forward simulator
        # (#4120) populates them, but the TDD oracle path uses a kinematic
        # mock; the default keeps the driver usable in both regimes. Callers
        # that ship a torque-aware simulator may override to ``total_work``
        # to match the Simscape baseline penalty.
        default_factory=lambda: CostOptions(regularizer="coeff_l2", lambda_=1e-6)
    )
    verbose: bool = False


# --------------------------------------------------------------------------- #
# Argument validation (DbC: enforce the canonical contract on inputs)
# --------------------------------------------------------------------------- #


def _resolve_n_joints(opts: FitOptions) -> int:
    """Resolve ``n_joints`` from ``opts`` or the simulator hint."""
    if opts.n_joints is not None:
        if opts.n_joints <= 0:
            raise ValueError(
                f"FitOptions.n_joints must be positive, got {opts.n_joints}"
            )
        return int(opts.n_joints)
    sim_fn = opts.simulate_fn
    inferred = getattr(sim_fn, "n_joints", None) if sim_fn is not None else None
    if inferred is not None:
        return int(inferred)
    return DEFAULT_N_JOINTS


def _resolve_simulate_fn(
    opts: FitOptions,
) -> Callable[[NDArray[np.float64]], SimOutput]:
    """Return the simulate function to use, or import the OpenSim default.

    Raises:
        RuntimeError: If no simulate function is supplied and the OpenSim
            ``simulate_with_coefficients`` wrapper (issue #4120) is not
            installed in the workspace.
    """
    if opts.simulate_fn is not None:
        if not callable(opts.simulate_fn):
            raise TypeError(
                "FitOptions.simulate_fn must be callable, got "
                f"{type(opts.simulate_fn).__name__}"
            )
        return opts.simulate_fn

    try:  # pragma: no cover - exercised only when #4120 has landed
        from .simulate import simulate_with_coefficients
    except ImportError as exc:  # pragma: no cover - default path until #4120
        raise RuntimeError(
            "fit_swing_opensim requires either an explicit "
            "FitOptions.simulate_fn or the OpenSim "
            "simulate_with_coefficients wrapper from issue #4120. "
            "Install/merge #4120 or pass a simulate_fn for tests."
        ) from exc

    return simulate_with_coefficients  # type: ignore[return-value]


def _resolve_theta0(
    opts: FitOptions, lb: NDArray[np.float64], ub: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Resolve the warm-start vector. Deterministic given ``rng_seed``."""
    d = lb.size
    if opts.theta0 is not None:
        theta0 = np.asarray(opts.theta0, dtype=np.float64).reshape(-1)
        if theta0.size != d:
            raise ValueError(
                f"FitOptions.theta0 length {theta0.size} != n_joints*7 = {d}"
            )
        if not np.all(np.isfinite(theta0)):
            raise ValueError("FitOptions.theta0 must be finite")
        # Clip to bounds so SLSQP does not start at a corner outside the box.
        return np.clip(theta0, lb, ub).astype(np.float64, copy=False)

    rng = np.random.default_rng(opts.rng_seed)
    # Sample within 5% of zero so the warm start does not begin in a
    # divergent corner of the bound box (mirrors the Simscape reference).
    scale = 0.05
    return rng.uniform(scale * lb, scale * ub).astype(np.float64)


def _check_target(target: ClubTarget) -> None:
    """ClubTarget validates itself at construction; assert that here."""
    if not isinstance(target, ClubTarget):
        raise TypeError(
            f"target must be a ClubTarget instance, got {type(target).__name__}"
        )


# --------------------------------------------------------------------------- #
# Public driver
# --------------------------------------------------------------------------- #


def fit_swing_opensim(
    target: ClubTarget,
    options: FitOptions | None = None,
) -> FitResult:
    """Fit polynomial torque coefficients to ``target`` using SLSQP.

    The optimizer minimises the canonical
    :func:`src.shared.python.motion_matching.cost.compute_cost` against the
    forward simulator supplied by ``options.simulate_fn`` (or the OpenSim
    ``simulate_with_coefficients`` wrapper from issue #4120).

    Args:
        target:  Validated :class:`ClubTarget` (loader-produced; the dataclass
                 enforces shape/finite/quaternion-norm contracts).
        options: :class:`FitOptions`. Defaults are spec-faithful.

    Returns:
        :class:`FitResult` with the recovered ``theta``, final cost,
        iteration / eval counts, success flag, ``solver_status``, message,
        elapsed wall-clock, per-iteration history, and method name.

    Raises:
        TypeError:  If inputs do not match the declared types.
        ValueError: If ``options.theta0`` length disagrees with
                    ``n_joints * 7`` or the simulate function returns a
                    badly-shaped :class:`SimOutput`.
        RuntimeError: If no simulate function is supplied and the OpenSim
                    forward-sim wrapper (issue #4120) is unavailable.
    """
    from scipy.optimize import minimize  # heavy import; defer

    if options is None:
        options = FitOptions()

    _check_target(target)
    n_joints = _resolve_n_joints(options)
    d = n_joints * POLY_ORDER_PER_JOINT

    bound = float(options.coeff_bound)
    if not (np.isfinite(bound) and bound > 0):
        raise ValueError(
            f"FitOptions.coeff_bound must be a positive finite scalar, got {bound}"
        )
    lb = np.full(d, -bound, dtype=np.float64)
    ub = np.full(d, +bound, dtype=np.float64)

    simulate_fn = _resolve_simulate_fn(options)
    theta0 = _resolve_theta0(options, lb, ub)

    history: list[float] = []
    n_eval = {"count": 0}

    def J(theta: NDArray[np.float64]) -> float:
        n_eval["count"] += 1
        try:
            cost, _ = compute_cost(
                np.asarray(theta, dtype=np.float64),
                target,
                simulate_fn,
                options.cost_options,
            )
        except Exception:  # noqa: BLE001 - logged + re-raised
            logger.exception("compute_cost raised at iter=%d", n_eval["count"])
            raise
        history.append(cost)
        if options.verbose:
            logger.info("fit_swing_opensim iter=%d J=%.6e", n_eval["count"], cost)
        return cost

    bounds: list[tuple[float, float]] = list(zip(lb.tolist(), ub.tolist(), strict=True))

    t0 = time.perf_counter()
    res: Any = minimize(
        J,
        theta0,
        method=options.method,
        bounds=bounds,
        options={"maxiter": options.max_iter, "ftol": options.ftol},
    )
    elapsed = time.perf_counter() - t0

    success = bool(res.success)
    if success:
        status = "success"
    elif np.isfinite(float(res.fun)):
        status = "warning"
    else:
        status = "failed"

    # Spec §2.2: post-fit ``theta_optimal`` must satisfy length+finiteness
    # before downstream code consumes it. The optimizer's box bounds are
    # symmetric ``[-coeff_bound, +coeff_bound]`` per letter, so we pass
    # the same bounds to surface any infeasible solver step.
    bound_table = {
        chr(ord("A") + col): (-bound, bound) for col in range(POLY_ORDER_PER_JOINT)
    }
    theta_opt = validate_theta(
        np.asarray(res.x, dtype=np.float64),
        n_joints=n_joints,
        bounds=bound_table,
        name="theta_optimal",
    )

    result = FitResult(
        theta_optimal=theta_opt.copy(),
        final_cost=float(res.fun),
        final_rmse_m=float("nan"),
        solver_status=status,
        iterations=int(getattr(res, "nit", 0)),
        n_evaluations=int(n_eval["count"]),
        wall_clock_s=float(elapsed),
        message=str(res.message),
        history=tuple(history),
        method=str(options.method),
        git_commit="unknown",
        engine_version="unknown",
        target_hash="unknown",
        timestamp_utc="unknown",
    )

    # Postcondition: history should be non-empty and end at the reported cost
    # within floating-point tolerance.
    assert result.history, "FitResult.history must not be empty"
    assert np.isfinite(result.final_cost), "FitResult.cost must be finite"
    assert result.theta_optimal.shape == (d,), (
        f"FitResult.theta shape {result.theta_optimal.shape} != ({d},)"
    )
    return result
