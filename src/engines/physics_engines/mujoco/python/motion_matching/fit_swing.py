"""Canonical MuJoCo fit driver for motion matching.

Implements ``fit_swing_mujoco`` per
``src/engines/physics_engines/mujoco/MUJOCO_PARITY_SPEC.md`` §2.2 and the
cross-engine spec at ``src/engines/CROSS_ENGINE_PARITY_SPEC.md`` §2.2.

Public API:
    FitOptions  -- frozen dataclass holding CostOptions, SimOptions,
                   MinimizerOptions, rng_seed.
    MinimizerOptions -- scipy ``minimize`` knobs in one place.
    FitResult   -- frozen dataclass mirroring the Simscape provenance block.
    fit_swing_mujoco(target, options) -> FitResult.

Speed
-----
MuJoCo's killer feature is fast forward sim (~3 ms per 0.3 s rollout on a
17-DOF model — see MUJOCO_PARITY_SPEC.md §6.1). For the MVP we drive
``scipy.optimize.minimize(method='SLSQP')`` with element-wise box bounds and
no analytic Jacobian; the SQP loop converges in ~20 iterations on the
synth-then-fit oracle.

The follow-up to wire MuJoCo's ``mjd_transitionFD`` analytic state-transition
Jacobian into the gradient is tracked separately (see issue tagged
``mujoco`` + ``priority:medium``); chaining ``dx/du`` through the
polynomial-driver derivative ``du/dθ`` (closed-form: powers of t) is the
path from < 5 s/fit to the spec target of < 0.5 s/fit.

Threading
---------
``mjcb_control`` is a process-global callback in MuJoCo. Each ``fit`` call
serially evaluates ``simulate_with_coefficients``, which installs and
uninstalls the driver per rollout, so back-to-back fits in the same process
are safe. Parallel fits MUST use ``multiprocessing`` (not threads).
"""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.api.utils.datetime_compat import UTC
from src.shared.python.core.contracts.decorators import (
    postcondition,
    precondition,
)
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.cost import (
    CostOptions,
    SimOutput,
    compute_cost,
    compute_total_work,
)
from src.shared.python.motion_matching.fit_result import CanonicalFitResult as FitResult
from src.shared.python.motion_matching.validate_theta import validate_theta

from .jacobians import JacobianCache, compute_cost_gradient_analytical
from .simulate import SimOptions, SimOut, simulate_with_coefficients
from .torque_driver import polynomial_torque_bounds

__all__ = [
    "FitOptions",
    "FitResult",
    "MinimizerOptions",
    "fit_swing_mujoco",
]

SolverName = Literal["SLSQP", "L-BFGS-B"]
JacMode = Literal["finite_difference", "analytical"]


# --- Options -----------------------------------------------------------------


@dataclass(frozen=True)
class MinimizerOptions:
    """scipy ``minimize`` knobs.

    Attributes:
        method:    SLSQP (default, supports box bounds) or L-BFGS-B.
        maxiter:   iteration cap. Per spec §6.2 the budget is 20 for the
                   synth-recovery oracle; the default of 200 is the
                   Simscape parity number.
        ftol:      objective tolerance. Mirrors the Simscape default.
        theta0:    optional warm-start (length ``n_joints * 7``); when
                   ``None``, sample uniformly from a small fraction of the
                   bound box (see ``warm_start_scale``).
        warm_start_scale: when ``theta0 is None``, sample
                   ``rng.uniform(scale*lb, scale*ub)``. 0.05 is small
                   enough that scipy doesn't start at a divergent corner.
    """

    method: SolverName = "SLSQP"
    maxiter: int = 200
    ftol: float = 1e-6
    theta0: NDArray[np.float64] | None = None
    warm_start_scale: float = 0.05
    jac_mode: JacMode = "finite_difference"


@dataclass(frozen=True)
class FitOptions:
    """Top-level fit options per MUJOCO_PARITY_SPEC §5.1.

    Each sub-options dataclass is frozen, so the entire ``FitOptions`` graph
    is hashable / pure.
    """

    cost: CostOptions = field(default_factory=CostOptions)
    sim: SimOptions = field(default_factory=SimOptions)
    minimizer: MinimizerOptions = field(default_factory=MinimizerOptions)
    rng_seed: int = 0

    # ---- LOD-2 accessors (CLAUDE.md §LOD) ----
    @property
    def maxiter(self) -> int:
        """Delegating accessor — keeps callers off ``opts.minimizer.maxiter``."""
        return self.minimizer.maxiter

    @property
    def method(self) -> SolverName:
        return self.minimizer.method

    @property
    def ftol(self) -> float:
        return self.minimizer.ftol


# --- Provenance helpers ------------------------------------------------------


def _hash_target(target: ClubTarget) -> str:
    """SHA-256 of the canonical bytes of a ``ClubTarget``.

    The hash MUST be stable for identical numeric content, so we serialize
    the contiguous float64 buffers in a fixed order, not the Python repr.
    """
    h = hashlib.sha256()
    for arr in (target.time, target.butt, target.clubhead, target.club_quat):
        a = np.ascontiguousarray(arr, dtype=np.float64)
        h.update(a.tobytes())
    h.update(int(target.impact_idx).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def _git_commit_short() -> str:
    """Return the short SHA of HEAD, or ``"unknown"`` if not in a repo.

    Delegates to the shared probe (issue #6939) so the git-commit logic is
    not reimplemented per engine.
    """
    from src.shared.python.motion_matching.provenance import git_commit_short

    return git_commit_short()


def _mujoco_version() -> str:
    """Return ``mujoco.__version__`` or a marker if unavailable."""
    try:
        import mujoco

        return str(getattr(mujoco, "__version__", "unknown"))
    except ImportError:
        return "unavailable"


# --- Internal: forward sim adapter ------------------------------------------


def _simulate_for_cost(
    theta: NDArray[np.float64],
    sim_opts: SimOptions,
    target: ClubTarget,
) -> SimOutput:
    """Run the MuJoCo rollout and adapt the output to ``cost.SimOutput``.

    The shared cost function expects ``N`` rows matching ``target.time``;
    if the rollout returned a different ``N`` (because ``output_rate_hz``
    or ``T_s`` were misconfigured) we raise here so the optimizer fails
    fast rather than silently mis-aligning samples.
    """
    out: SimOut = simulate_with_coefficients(theta, sim_opts)
    n_target = target.time.shape[0]
    if out.time.shape[0] != n_target:
        raise ValueError(
            f"sim returned {out.time.shape[0]} frames but target has "
            f"{n_target}; check FitOptions.sim.T_s and output_rate_hz "
            f"against the target's time grid"
        )
    return out.to_cost_simoutput()


def _final_rmse_m(sim_out: SimOutput, target: ClubTarget) -> float:
    """sqrt-mean of (||butt-diff||^2 + ||ch-diff||^2) — RMS metres at fit."""
    db = sim_out.butt - target.butt
    dc = sim_out.clubhead - target.clubhead
    # ⚡ Bolt: np.vdot is ~3-4x faster than np.sum(x*x, axis=1)
    # and avoids temporary allocations
    return float(np.sqrt((np.vdot(db, db) + np.vdot(dc, dc)) / db.shape[0]))


# --- Warm start --------------------------------------------------------------


def _warm_start_theta(
    minimizer: MinimizerOptions,
    lb: NDArray[np.float64],
    ub: NDArray[np.float64],
    rng_seed: int,
) -> NDArray[np.float64]:
    """Return a length-``len(lb)`` warm-start vector inside the bounds."""
    d = lb.size
    if minimizer.theta0 is not None:
        theta0 = np.asarray(minimizer.theta0, dtype=np.float64).reshape(-1)
        if theta0.size != d:
            raise ValueError(
                f"theta0 has length {theta0.size}, expected {d} (= n_joints*7)"
            )
        if not np.all(np.isfinite(theta0)):
            raise ValueError("theta0 must be finite")
        return np.clip(theta0, lb, ub)
    rng = np.random.default_rng(rng_seed)
    scale = float(minimizer.warm_start_scale)
    if not (0.0 < scale <= 1.0):
        raise ValueError(
            f"warm_start_scale must be in (0, 1]; got {scale}; outside the "
            "bound box scipy may diverge before the first iteration"
        )
    return rng.uniform(scale * lb, scale * ub).astype(np.float64)


# --- Precondition / postcondition predicates --------------------------------


def _check_args(target: ClubTarget, options: FitOptions) -> bool:
    return isinstance(target, ClubTarget) and isinstance(options, FitOptions)


def _check_result(result: FitResult) -> bool:
    return (
        isinstance(result, FitResult)
        and np.isfinite(result.final_rmse_m)
        and result.final_rmse_m >= 0.0
        and len(result.target_hash) == 64
    )


# --- Public entry point ------------------------------------------------------


@precondition(
    _check_args,
    "fit_swing_mujoco requires a ClubTarget target and a FitOptions options",
)
@postcondition(
    _check_result,
    "fit_swing_mujoco must return a FitResult with finite, non-negative RMSE "
    "and a 64-char target_hash",
)
def fit_swing_mujoco(target: ClubTarget, options: FitOptions) -> FitResult:
    """Fit polynomial-torque coefficients ``θ`` to a measured club trajectory.

    Args:
        target: validated :class:`ClubTarget` (CLUB_IK_SPEC schema). The
            shape of ``target.time`` defines the rollout grid; ``options.sim``
            must be configured to produce the same number of frames.
        options: :class:`FitOptions` aggregating cost, sim, minimizer
            settings, and the RNG seed for the warm-start draw.

    Returns:
        :class:`FitResult` with the recovered ``θ``, the final RMSE, the
        cost breakdown, and the provenance fields required by
        ``CODING_STANDARDS.md``.

    Raises:
        ValueError: on malformed ``options.minimizer.theta0`` or a
            sim/target frame mismatch.
        TypeError:  via the precondition if either argument is the wrong
            type.

    Performance
    -----------
    Per ``MUJOCO_PARITY_SPEC.md`` §6.2 the target is ``< 0.5 s`` per fit.
    The MVP uses scipy SLSQP with finite-difference gradients (no
    ``jac=`` callable). The follow-up to swap that for MuJoCo's analytic
    ``mjd_transitionFD`` Jacobian is tracked separately (see the
    ``mujoco`` + ``priority:medium`` issue list); the speed-up needed to
    hit the spec target is ~10×.
    """
    t_wall_start = time.perf_counter()
    timestamp_utc = datetime.now(UTC).isoformat()

    # --- 1. Compile the model once to discover ``n_joints``. ---------------
    # We can't compute bounds without ``model.nu``; the obvious way is to
    # call simulate once, but a degenerate first eval would waste a
    # rollout. Compile-and-discard is < 10 ms.
    n_joints = _discover_n_joints(options.sim)

    lb, ub = polynomial_torque_bounds(n_joints)
    bounds = list(zip(lb.tolist(), ub.tolist(), strict=True))

    theta0 = _warm_start_theta(options.minimizer, lb, ub, options.rng_seed)

    # --- 2. Objective with history tracking -------------------------------
    history: list[float] = []
    n_eval = 0

    sim_opts = options.sim

    def _sim_fn(theta: NDArray[np.float64]) -> SimOutput:
        return _simulate_for_cost(theta, sim_opts, target)

    def J(theta: NDArray[np.float64]) -> float:
        nonlocal n_eval
        n_eval += 1
        try:
            cost, _ = compute_cost(theta, target, _sim_fn, options.cost)
        except (ValueError, RuntimeError) as exc:
            # Soft-fail: a divergent rollout returns a large finite cost
            # so SLSQP can still take a step away from this region.
            history.append(float("inf"))
            raise RuntimeError(f"objective evaluation failed: {exc}") from exc
        history.append(float(cost))
        return float(cost)

    # --- 3. Drive scipy SLSQP --------------------------------------------
    from scipy.optimize import minimize  # heavy import; deferred until call

    scipy_options = {"maxiter": options.maxiter, "ftol": options.ftol}
    jac_mode = options.minimizer.jac_mode
    if jac_mode == "analytical":
        # Cache MJCF compile + scratch buffers across the whole fit.
        jac_cache = JacobianCache()

        def grad_J(theta_in: NDArray[np.float64]) -> NDArray[np.float64]:
            return compute_cost_gradient_analytical(
                theta_in, target, sim_opts, options.cost, cache=jac_cache
            )

        # scipy-stubs' ``minimize`` overloads do not cover these
        # (callable, ndarray, method, jac/bounds, options=dict) forms.
        res = minimize(  # type: ignore[call-overload]
            J,
            theta0,
            method=options.method,
            jac=grad_J,
            bounds=bounds,
            options=scipy_options,
        )
    elif jac_mode == "finite_difference":
        res = minimize(  # type: ignore[call-overload]
            J,
            theta0,
            method=options.method,
            bounds=bounds,
            options=scipy_options,
        )
    else:
        raise ValueError(
            f"unknown jac_mode {jac_mode!r}; expected "
            "'finite_difference' or 'analytical'"
        )

    # --- 4. Re-evaluate the optimum to get a clean SimOutput -------------
    # Spec §2.2: post-fit ``theta_optimal`` must satisfy length+finiteness
    # before we hand it to the next forward-sim. SLSQP can return
    # ``inf`` / ``nan`` on a hard solver failure; surfacing that here as
    # a ``ValueError`` is preferable to a silent NaN trajectory.
    theta_star = validate_theta(
        np.asarray(res.x, dtype=np.float64),
        n_joints=n_joints,
        name="theta_optimal",
    )
    final_sim_out = _sim_fn(theta_star)
    final_cost, breakdown = compute_cost(
        theta_star, target, lambda _t: final_sim_out, options.cost
    )
    rmse_m = _final_rmse_m(final_sim_out, target)
    work_J = compute_total_work(final_sim_out)

    duration_s = time.perf_counter() - t_wall_start
    solver_options = {
        **scipy_options,
        "warm_start_scale": options.minimizer.warm_start_scale,
        "rng_seed": options.rng_seed,
        "jac_mode": jac_mode,
        "platform": platform.platform(),
    }

    return FitResult(
        theta_optimal=theta_star,
        final_cost=final_cost,
        final_rmse_m=rmse_m,
        solver_status=(
            "success"
            if bool(res.success)
            else ("warning" if "iteration" in str(res.message).lower() else "failed")
        ),
        iterations=int(getattr(res, "nit", 0)),
        n_evaluations=n_eval,
        wall_clock_s=duration_s,
        message=str(res.message),
        history=tuple(history),
        method=options.method,
        git_commit=_git_commit_short(),
        engine_version=_mujoco_version(),
        target_hash=_hash_target(target),
        timestamp_utc=timestamp_utc,
        cost_breakdown=breakdown,
        final_total_work_J=work_J,
        solver_options=solver_options,
    )


# --- Helpers ----------------------------------------------------------------


def _discover_n_joints(sim_opts: SimOptions) -> int:
    """Compile the MJCF for ``sim_opts.variant`` and return ``model.nu``."""
    import mujoco

    # DRY: mirror simulate._load_model_xml without exposing a private helper.
    if sim_opts.variant == "full":
        from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
            FULL_BODY_GOLF_SWING_XML as _xml,
        )
    elif sim_opts.variant == "upper":
        from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
            UPPER_BODY_GOLF_SWING_XML as _xml,
        )
    elif sim_opts.variant == "advanced":
        from src.engines.physics_engines.mujoco._golf_swing_advanced_xml import (
            ADVANCED_BIOMECHANICAL_GOLF_SWING_XML as _xml,
        )
    else:
        raise ValueError(
            f"unknown variant {sim_opts.variant!r}; expected upper/full/advanced"
        )
    return int(mujoco.MjModel.from_xml_string(_xml).nu)
