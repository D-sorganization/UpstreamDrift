"""Python-driven optimization for Option 4.

``fit_swing_scipy`` runs ``scipy.optimize.minimize(method="SLSQP")`` against
the Simscape forward simulation provided by :class:`SimscapeAdapter`. This is
the Python-side analogue of Option 1's ``fit_swing_fmincon.m``.

JAX support is left for a follow-up issue (#4075 — Option 2 surrogate); a
JAX-flavoured fit needs a differentiable forward model, which the MATLAB
engine does not provide. Once the surrogate ships, ``fit_swing_jax`` will be
a thin wrapper around ``jax.scipy.optimize.minimize`` over the surrogate's
forward + JIT-compiled cost.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # support both package-style and tests/conftest.py path-injection import
    from .simscape_adapter import ClubTarget, SimscapeAdapter
except ImportError:  # pragma: no cover - fallback for ad-hoc imports
    from simscape_adapter import ClubTarget, SimscapeAdapter  # type: ignore[no-redef]

from src.shared.python.motion_matching.multi_source_target import MultiSourceTarget

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class FitOptions:
    """Options for ``fit_swing_scipy``.

    Attributes:
        method: scipy method name. ``"SLSQP"`` (default) handles bounds,
            ``"L-BFGS-B"`` is a fast unconstrained-with-bounds alternative.
        max_iter: optimizer iteration cap.
        ftol: scipy ``ftol`` for the objective.
        theta0: optional warm-start vector (length ``n_joints*7``); when
            ``None``, a uniform-random sample within the bounds is used.
        rng_seed: RNG seed for the warm-start draw.
        cost_opts: optional overrides for ``default_cost_options`` on the
            MATLAB side (e.g. ``{"lambda": 0.0}``).
        verbose: log per-iteration cost when True.
    """

    method: str = "SLSQP"
    max_iter: int = 200
    ftol: float = 1e-6
    theta0: np.ndarray | None = None
    rng_seed: int = 42
    cost_opts: dict[str, Any] = field(default_factory=dict)
    verbose: bool = False


@dataclass
class FitResult:
    """Result of one ``fit_swing_scipy`` run."""

    theta: np.ndarray
    cost: float
    n_iter: int
    n_eval: int
    success: bool
    message: str
    elapsed_s: float
    history: list[float]
    method: str


# --------------------------------------------------------------------------- #
# scipy fit
# --------------------------------------------------------------------------- #


def fit_swing_scipy(
    target: MultiSourceTarget | ClubTarget,
    adapter: SimscapeAdapter,
    options: FitOptions | None = None,
) -> FitResult:
    """Fit polynomial torque coefficients to a target trajectory using scipy.

    Args:
        target: measured trajectory (see :class:`ClubTarget`).
        adapter: live :class:`SimscapeAdapter`. The engine must already be
            startable; ``adapter.start()`` is called if needed.
        options: optimizer options (defaults via :class:`FitOptions`).

    Returns:
        :class:`FitResult` with the recovered theta, final cost, and history.
    """
    from scipy.optimize import minimize  # heavy import; deferred until called

    if options is None:
        options = FitOptions()

    adapter.start()
    n_joints = adapter.get_n_joints()
    lb, ub = adapter.get_polynomial_bounds(n_joints)
    d = lb.size

    # ---------------- warm start ---------------- #
    if options.theta0 is not None:
        theta0 = np.asarray(options.theta0, dtype=np.float64).reshape(-1)
        if theta0.size != d:
            raise ValueError(f"theta0 has length {theta0.size}, expected {d}")
    else:
        rng = np.random.default_rng(options.rng_seed)
        # sample within 5% of zero so the optimizer does not start at a
        # hopelessly-divergent corner of the bound box.
        scale = 0.05
        theta0 = rng.uniform(scale * lb, scale * ub).astype(np.float64)

    # ---------------- objective ---------------- #
    history: list[float] = []
    n_eval = 0

    def J(theta: np.ndarray) -> float:
        nonlocal n_eval
        n_eval += 1
        cost = adapter.compute_cost(theta, target, opts=options.cost_opts)
        history.append(cost)
        if options.verbose:
            logger.info("fit_swing_scipy iter=%d J=%.6e", n_eval, cost)
        return cost

    bounds = list(zip(lb.tolist(), ub.tolist(), strict=False))

    # ---------------- run ---------------- #
    t0 = time.perf_counter()
    res = minimize(
        J,
        theta0,
        method=options.method,
        bounds=bounds,
        options={"maxiter": options.max_iter, "ftol": options.ftol},
    )
    elapsed = time.perf_counter() - t0

    return FitResult(
        theta=np.asarray(res.x, dtype=np.float64),
        cost=float(res.fun),
        n_iter=int(getattr(res, "nit", 0)),
        n_eval=n_eval,
        success=bool(res.success),
        message=str(res.message),
        elapsed_s=elapsed,
        history=history,
        method=options.method,
    )


# --------------------------------------------------------------------------- #
# JAX path — deferred to issue #4075
# --------------------------------------------------------------------------- #


def fit_swing_jax(  # pragma: no cover - explicitly unimplemented
    target: MultiSourceTarget | ClubTarget,
    adapter: SimscapeAdapter,
    options: FitOptions | None = None,
) -> FitResult:
    """JAX-driven fit over the Option-2 surrogate.

    TODO(#4075): Implement once the Option-2 surrogate (issue #4075) ships.
    The MATLAB Engine's forward simulator is not differentiable, so a JAX
    optimizer needs to drive the surrogate, not Simscape. The expected wiring
    is::

        from src.learning.surrogate.option2 import OptionTwoSurrogate
        surrogate = OptionTwoSurrogate.load(...)
        @jax.jit
        def cost(theta): ...
        sol = jaxopt.ScipyMinimize(fun=cost, method="L-BFGS-B").run(theta0)
        # then validate sol.params via adapter.simulate_with_coefficients

    Until then, callers should use :func:`fit_swing_scipy`.
    """
    raise NotImplementedError(
        "fit_swing_jax is deferred to issue #4075 (Option-2 surrogate). "
        "Use fit_swing_scipy until the differentiable surrogate lands."
    )
