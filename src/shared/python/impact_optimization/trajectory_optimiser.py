"""Trust-region impact optimiser around the matched swing (SHOW-C3, #14559).

Searches spline torque corrections ``delta`` added to the matched torque
``tau0``: ``tau = clip(tau0 + spline(delta), envelope)``.  Zero ``delta`` *is*
the matched swing, so the search starts at the fit and the trust region keeps
it there.  Three things are guaranteed by construction rather than by
penalty alone:

* torque never exceeds the :class:`PhysiologicalEnvelope` (projection), so a
  binding limit is *reported*, never silently exceeded;
* the returned solution lies inside the :class:`TrustRegion` (the correction
  is scaled back along the ray to the matched swing if needed);
* the result is deterministic for a fixed ``seed`` (the seed only drives the
  optional restart perturbations; the default start is the matched swing).

The plant is injected as ``rollout(torque_nm) -> RolloutOutcome`` so the same
solver runs on a reduced analytic model (tests) and on the full-body forward
dynamics (the C3 adapter).

Design by Contract:
    Preconditions: baseline torque ``(T, n)`` matches ``times_s`` and the
    envelope; trust-region reference matches the rollout trajectory shape.
    Postconditions: ``max_utilisation <= 1`` and
    ``deviation_from_fit <= trust_region.radius`` for every returned solution.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.shared.python.contracts import ensure, require
from src.shared.python.impact_optimization.impact_target import (
    ImpactTarget,
    MeasuredImpact,
)
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.spline_controls import SplineTorqueControl
from src.shared.python.impact_optimization.trust_region import TrustRegion


@dataclass(frozen=True)
class ImpactTolerance:
    """Acceptance tolerances on |impact - target| (issue #14559)."""

    face_deg: float = 0.5
    path_deg: float = 1.5
    aoa_deg: float = 1.0


@dataclass(frozen=True)
class OptimiserConfig:
    """Solver settings. ``restarts`` extra starts use ``seed`` (default none)."""

    n_knots: int = 6
    seed: int = 0
    max_iterations: int = 100
    restarts: int = 0
    restart_scale_nm: float = 5.0
    trust_penalty_weight: float = 1.0e3
    tolerance: ImpactTolerance = ImpactTolerance()

    def __post_init__(self) -> None:
        require(self.n_knots >= 2, "n_knots must be >= 2", self.n_knots)
        require(self.max_iterations >= 1, "max_iterations must be >= 1")
        require(self.restarts >= 0, "restarts must be >= 0")


@dataclass(frozen=True)
class RolloutOutcome:
    """What one forward simulation returns."""

    impact: MeasuredImpact
    torque_nm: np.ndarray
    trajectory: np.ndarray


Rollout = Callable[[np.ndarray], RolloutOutcome]


@dataclass(frozen=True, eq=False)
class ImpactSolution:
    """Optimised swing plus everything needed to audit it."""

    knots: np.ndarray
    outcome: RolloutOutcome
    deviation: float
    deviation_from_fit: float
    binding_joints: tuple[str, ...]
    max_utilisation: float
    envelope_limits_nm: np.ndarray
    tolerances_met: bool
    #: face - path (deg): sets curvature, so the 10:1 trade-off is auditable.
    face_to_path_deg: float = float("nan")

    @property
    def achieved_speed_mph(self) -> float:
        """Impact speed of the solved swing (delegates to the outcome)."""
        return float(self.outcome.impact.speed_mph)


def _within_tolerance(
    impact: MeasuredImpact, target: ImpactTarget, tol: ImpactTolerance
) -> bool:
    if impact.face_deg is None or impact.path_deg is None:
        return False
    return bool(
        abs(impact.face_deg - target.face_deg) <= tol.face_deg
        and abs(impact.path_deg - target.path_deg) <= tol.path_deg
        and abs(impact.aoa_deg - target.aoa_deg) <= tol.aoa_deg
    )


def face_to_path_deg(impact: MeasuredImpact) -> float:
    """face - path (deg); NaN when either is unobservable."""
    face, path = impact.face_deg, impact.path_deg
    if face is None or path is None:
        return math.nan
    return float(face - path)


def _project(tau: np.ndarray, env: PhysiologicalEnvelope) -> np.ndarray:
    return np.clip(tau, -env.tau_max_negative_nm, env.tau_max_positive_nm)


def solve_impact_trajectory(
    rollout: Rollout,
    baseline_torque_nm: np.ndarray,
    times_s: np.ndarray,
    target: ImpactTarget,
    envelope: PhysiologicalEnvelope,
    trust_region: TrustRegion,
    config: OptimiserConfig | None = None,
    initial_knots: np.ndarray | None = None,
) -> ImpactSolution:
    """Minimise weighted impact deviation inside the trust region.

    ``initial_knots`` (``n_knots x n_joints``) warm-starts the search, e.g. from
    the previous speed in a continuation; default is the matched swing.
    """
    cfg = config or OptimiserConfig()
    base = np.asarray(baseline_torque_nm, dtype=float)
    times = np.asarray(times_s, dtype=float)
    n = len(envelope.names)
    require(base.shape == (times.size, n), "baseline must be (T, n_joints)", base.shape)
    ctrl = SplineTorqueControl(np.linspace(times[0], times[-1], cfg.n_knots), n)

    def run(p: np.ndarray) -> RolloutOutcome:
        tau = _project(base + ctrl.torque(ctrl.unpack(p), times), envelope)
        return rollout(tau)

    def cost(p: np.ndarray) -> float:
        out = run(p)
        pen = trust_region.penalty(out.trajectory)
        return target.deviation(out.impact) + cfg.trust_penalty_weight * pen

    rng = np.random.default_rng(cfg.seed)
    if initial_knots is None:
        x0_first = np.zeros(ctrl.n_params)
    else:
        k0 = np.asarray(initial_knots, dtype=float)
        require(k0.shape == (cfg.n_knots, n), "initial_knots shape mismatch", k0.shape)
        x0_first = k0.ravel()
    starts = [x0_first] + [
        rng.normal(scale=cfg.restart_scale_nm, size=ctrl.n_params)
        for _ in range(cfg.restarts)
    ]
    best_p, best_c = starts[0], np.inf
    for x0 in starts:
        res = minimize(
            cost,
            x0,
            method="L-BFGS-B",
            options={
                "maxiter": cfg.max_iterations,
                "eps": 1e-3,
                "ftol": 1e-14,
                "gtol": 1e-10,
            },
        )
        if res.fun < best_c:
            best_p, best_c = np.asarray(res.x, dtype=float), float(res.fun)

    # Guarantee the trust region: shrink the correction toward the fit.
    if trust_region.deviation(run(best_p).trajectory) > trust_region.radius:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            inside = trust_region.deviation(run(mid * best_p).trajectory)
            lo, hi = (mid, hi) if inside <= trust_region.radius else (lo, mid)
        best_p = lo * best_p

    out = run(best_p)
    util = np.max(
        np.stack([envelope.utilisation(row) for row in out.torque_nm]), axis=0
    )
    binding = tuple(
        name for name, u in zip(envelope.names, util, strict=True) if u >= 0.98
    )
    sol = ImpactSolution(
        knots=ctrl.unpack(best_p),
        outcome=out,
        deviation=target.deviation(out.impact),
        deviation_from_fit=trust_region.deviation(out.trajectory),
        binding_joints=binding,
        max_utilisation=float(np.max(util)),
        envelope_limits_nm=np.array(envelope.tau_max_positive_nm),
        tolerances_met=_within_tolerance(out.impact, target, cfg.tolerance),
        face_to_path_deg=face_to_path_deg(out.impact),
    )
    ensure(sol.max_utilisation <= 1.0 + 1e-9, "envelope exceeded", sol.max_utilisation)
    ensure(
        sol.deviation_from_fit <= trust_region.radius * (1 + 1e-9),
        "left the trust region",
        sol.deviation_from_fit,
    )
    return sol
