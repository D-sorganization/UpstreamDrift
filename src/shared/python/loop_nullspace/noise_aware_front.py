"""Pareto fronts of face-angle variance (NSX-92, #12721).

* :func:`variance_effort_front` - variance against capacity effort at a fixed
  nominal motion.  With M1 only the problem is convex and the front is the NSX-25
  epsilon-constraint front (:func:`~.pareto.pareto_front`) with the *effort* as
  the strictly convex primary and the variance as the exact second-order-cone
  bound, so every point is properly Pareto optimal.  With M2 the variance is not
  convex: the front is a sweep of effort caps, each point a certified local
  optimum (``certificate == "local"``) and the front is *not* claimed to be the
  true Pareto set.
* :func:`chs_variance_front` - variance against clubhead speed (CHS) over a
  family of cases.  CHS is determined by the nominal motion and is
  allocation-invariant, so it cannot be traded against variance by allocating;
  the front is the non-dominated set of per-case minimum variances at a common
  effort cap (V4), reusing the NSX-25 dominance rule.

Non-claim: conditional on the declared mechanisms and amplitudes; a point is
Pareto optimal among admissible allocations, not a recommendation (V8).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.allocation import _validated_frames
from src.shared.python.loop_nullspace.allocation_spec import ObjectiveSpec
from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.noise_aware_allocation import (
    _normalised_variance_term,
    minimise_face_variance,
)
from src.shared.python.loop_nullspace.noise_aware_model import NoiseAwareModel
from src.shared.python.loop_nullspace.noise_aware_spec import (
    NON_CLAIM,
    NoiseAwareResult,
    NoiseAwareSettings,
    NoiseAwareStatus,
)
from src.shared.python.loop_nullspace.pareto import (
    ParetoFront,
    ParetoPoint,
    _mark_dominated,
    pareto_front,
)
from src.shared.python.loop_nullspace.solvers import SolveStatus
from src.shared.python.loop_nullspace.variance_records import VarianceDeclaration

__all__ = [
    "ChsCase",
    "ChsFront",
    "ChsPoint",
    "FrontPoint",
    "VarianceEffortFront",
    "chs_variance_front",
    "variance_effort_front",
]

CHS_ALLOCATION_DEPENDENCE = "INVARIANT"


@dataclass(frozen=True)
class FrontPoint:
    """One non-trivially solved point: total variance against effort.

    ``certificate`` is ``"global-convex"`` (M1, solver KKT) or ``"local"``
    (M2, first-order stationary only).  ``slope`` is ``d effort / d variance``
    (convex fronts only).  On a non-solved ``status`` only ``status``,
    ``reason`` and ``target_variance`` are meaningful.
    """

    target_variance: float
    status: NoiseAwareStatus
    reason: str = ""
    variance: float | None = None
    effort: float | None = None
    slope: float | None = None
    certificate: str = ""
    z: tuple[Array, ...] = ()
    torque: Array | None = None
    dominated: bool = False


@dataclass(frozen=True)
class VarianceEffortFront:
    """Variance-versus-effort front with its floor and baseline.

    ``variance_floor`` is the minimum attainable variance (global for M1, best
    certified local for M2); ``baseline_variance`` / ``baseline_effort`` are
    those of the minimum-effort allocation.  ``pareto`` holds the raw NSX-25
    front when the problem is convex.
    """

    points: tuple[FrontPoint, ...]
    variance_floor: float
    baseline_variance: float
    baseline_effort: float
    convex: bool
    hypotheses: tuple[str, ...]
    pareto: ParetoFront | None = None
    non_claim: str = NON_CLAIM

    @property
    def solved(self) -> tuple[FrontPoint, ...]:
        return tuple(p for p in self.points if p.variance is not None)

    def is_monotone(self, rtol: float = 1e-8) -> bool:
        """Effort non-increasing in the allowed variance (convex fronts)."""
        pts = sorted(self.solved, key=lambda p: p.target_variance)
        eff = [float(p.effort) for p in pts if p.effort is not None]
        return all(
            b <= a + rtol * (1.0 + abs(a)) for a, b in zip(eff, eff[1:], strict=False)
        )


def _grid(floor: float, baseline: float, n_points: int) -> Array:
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if not baseline > floor:
        raise ValueError(
            "no trade-off: the baseline variance does not exceed the floor"
        )
    return np.asarray(floor + np.linspace(0.02, 1.0, n_points) * (baseline - floor))


def _convex_front(
    frames: tuple[Any, ...],
    ctx: dict[str, Any],
    targets: Array,
    model: NoiseAwareModel,
) -> tuple[tuple[FrontPoint, ...], ParetoFront]:
    times, scale = ctx["times"], ctx["scale"]
    coeff = model.signal_coefficients(np.zeros(model.n_z))
    term = _normalised_variance_term(coeff, scale)
    invariant = model.invariant_variance(np.zeros(model.n_z))
    effort = ObjectiveSpec(times_s=times, torque_scale=scale, effort_weight=1.0)
    bound = ObjectiveSpec(
        times_s=times, torque_scale=scale, effort_weight=0.0, variance=term
    )
    eps = term.weight * (targets - invariant)
    front = pareto_front(frames, (effort, bound), eps)
    points = []
    for tgt, pt in zip(targets, front.points, strict=True):
        if pt.status is not SolveStatus.OPTIMAL:
            points.append(
                FrontPoint(float(tgt), NoiseAwareStatus.INFEASIBLE, pt.reason)
            )
            continue
        if pt.primary is None or pt.constrained is None:
            raise ValueError("an optimal Pareto point must carry its objectives")
        slope = None if pt.slope is None else float(pt.slope * term.weight)
        points.append(
            FrontPoint(
                target_variance=float(tgt),
                status=NoiseAwareStatus.OPTIMAL,
                variance=float(invariant + pt.constrained / term.weight),
                effort=float(pt.primary),
                slope=slope,
                certificate="global-convex",
                z=pt.z,
                torque=pt.torque,
                dominated=pt.dominated,
            )
        )
    return tuple(points), front


def _need(value: float | None) -> float:
    """Return ``value`` as a float; raise if the solve did not provide it."""
    if value is None:
        raise ValueError("the allocation result lacks a required reference value")
    return float(value)


def _local_front(
    frames: tuple[Any, ...],
    ctx: dict[str, Any],
    targets: Array,
    floor: NoiseAwareResult,
) -> tuple[FrontPoint, ...]:
    lo, hi = _need(floor.effort), _need(floor.reference_effort)
    caps = np.linspace(hi, lo, targets.size)  # large cap -> small variance
    out = []
    for tgt, cap in zip(targets, caps, strict=True):
        res = minimise_face_variance(
            frames,
            ctx["times"],
            ctx["scale"],
            ctx["model"],
            ctx["decl"],
            NoiseAwareSettings(effort_cap=float(cap), starts=ctx["starts"]),
        )
        if res.variance is None:
            out.append(FrontPoint(float(tgt), res.status, res.reason))
            continue
        out.append(
            FrontPoint(
                float(tgt),
                res.status,
                variance=res.variance,
                effort=res.effort,
                certificate="local",
                z=res.z,
                torque=res.torque,
            )
        )
    return _mark_local_dominance(out)


def _mark_local_dominance(points: list[FrontPoint]) -> tuple[FrontPoint, ...]:
    """Dominance of local points via the NSX-25 rule (variance, effort both min)."""
    wrapped = [
        ParetoPoint(
            eps=0.0,
            status=SolveStatus.OPTIMAL
            if p.variance is not None
            else SolveStatus.NUMERICAL,
            primary=p.variance,
            constrained=p.effort,
        )
        for p in points
    ]
    marked = _mark_dominated(wrapped)
    return tuple(
        replace(p, dominated=m.dominated) for p, m in zip(points, marked, strict=True)
    )


def variance_effort_front(
    feasible_sets: Sequence[Any],
    times_s: Any,
    torque_scale: Any,
    model: NoiseAwareModel,
    declaration: VarianceDeclaration,
    n_points: int = 9,
    *,
    starts: tuple[Array, ...] = (),
) -> VarianceEffortFront:
    """Variance-versus-effort front between the variance floor and the baseline.

    Preconditions: those of :func:`~.noise_aware_allocation.minimise_face_variance`;
    ``n_points >= 2``; the variance depends on the allocation (a flat objective
    raises ``ValueError``: there is no trade-off to show, V1).  Postconditions:
    ``variance_floor <= every point variance <= baseline_variance`` (up to the
    solver tolerance); a convex front is monotone (``is_monotone``) and has no
    dominated solved point; a non-convex front is labelled ``local`` and
    ``convex`` is ``False``.
    """
    times, scale = np.asarray(times_s, float), np.asarray(torque_scale, float)
    frames = _validated_frames(
        tuple(feasible_sets),
        ObjectiveSpec(times_s=times, torque_scale=scale, effort_weight=1.0),
    )
    floor = minimise_face_variance(frames, times, scale, model, declaration)
    if floor.status is NoiseAwareStatus.OBJECTIVE_FLAT_INVARIANT:
        raise ValueError(
            "the variance does not depend on the allocation (V1): no front exists"
        )
    if floor.variance is None:
        raise ValueError(f"cannot build a front: {floor.status.value}: {floor.reason}")
    targets = _grid(floor.variance, _need(floor.reference_variance), n_points)
    ctx = {
        "times": times,
        "scale": scale,
        "model": model,
        "decl": declaration,
        "starts": starts,
    }
    if model.preload_active:
        points, raw, convex = _local_front(frames, ctx, targets, floor), None, False
    else:
        points, raw = _convex_front(frames, ctx, targets, model)
        convex = True
    return VarianceEffortFront(
        points=points,
        variance_floor=float(floor.variance),
        baseline_variance=_need(floor.reference_variance),
        baseline_effort=_need(floor.reference_effort),
        convex=convex,
        hypotheses=floor.hypotheses,
        pareto=raw,
    )


# ---------------------------------------------------------------------- CHS
@dataclass(frozen=True)
class ChsCase:
    """One nominal motion: its CHS, feasible sets and variance model.

    ``chs_mps`` is the clubhead speed of the case's nominal motion (an
    allocation-invariant quantity supplied by the caller from the motion, in
    m/s).  All cases of a front share the torque scale and ``times_s`` length.
    """

    label: str
    chs_mps: float
    feasible_sets: tuple[Any, ...]
    model: NoiseAwareModel

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a case needs a label")
        if not (math.isfinite(self.chs_mps) and self.chs_mps > 0.0):
            raise ValueError(f"chs_mps must be positive and finite, got {self.chs_mps}")
        object.__setattr__(self, "feasible_sets", tuple(self.feasible_sets))


@dataclass(frozen=True)
class ChsPoint:
    """Minimum variance of one case at the common effort cap."""

    label: str
    chs_mps: float
    status: NoiseAwareStatus
    reason: str = ""
    variance: float | None = None
    effort: float | None = None
    certificate: str = ""
    dominated: bool = False
    chs_allocation_dependence: str = CHS_ALLOCATION_DEPENDENCE


@dataclass(frozen=True)
class ChsFront:
    """Variance-versus-CHS points at a matched effort cap (V4)."""

    effort_cap: float
    points: tuple[ChsPoint, ...]
    hypotheses: tuple[str, ...]
    non_claim: str = NON_CLAIM

    @property
    def nondominated(self) -> tuple[ChsPoint, ...]:
        return tuple(
            p for p in self.points if p.variance is not None and not p.dominated
        )


def chs_variance_front(
    cases: Sequence[ChsCase],
    times_s: Any,
    torque_scale: Any,
    declaration: VarianceDeclaration,
    effort_cap: float,
) -> ChsFront:
    """Minimum variance per case at a common effort cap, with CHS dominance.

    Preconditions: at least two cases with distinct labels and ``effort_cap >
    0``.  Postconditions: every solved point has ``effort <= effort_cap`` (up to
    solver tolerance); a case whose minimum effort exceeds the cap is reported
    ``INFEASIBLE``, never relaxed; CHS is carried as ``INVARIANT``; point ``a``
    dominates ``b`` iff it has no lower CHS, no higher variance and is strictly
    better in one (NSX-25 rule on ``(variance, -CHS)``).
    """
    cs = tuple(cases)
    if len(cs) < 2 or not all(isinstance(c, ChsCase) for c in cs):
        raise ValueError("at least two ChsCase entries are required")
    if len({c.label for c in cs}) != len(cs):
        raise ValueError("case labels must be unique")
    if not (math.isfinite(effort_cap) and effort_cap > 0.0):
        raise ValueError(f"effort_cap must be positive and finite, got {effort_cap}")
    times, scale = np.asarray(times_s, float), np.asarray(torque_scale, float)
    points: list[ChsPoint] = []
    hyps: list[str] = []
    for case in cs:
        res = minimise_face_variance(
            case.feasible_sets,
            times,
            scale,
            case.model,
            declaration,
            NoiseAwareSettings(effort_cap=float(effort_cap)),
        )
        hyps += list(res.hypotheses)
        cert = (
            "local"
            if res.status is NoiseAwareStatus.LOCAL_STATIONARY
            else "global-convex"
        )
        if res.variance is None:
            points.append(ChsPoint(case.label, case.chs_mps, res.status, res.reason))
        else:
            points.append(
                ChsPoint(
                    case.label,
                    case.chs_mps,
                    res.status,
                    variance=res.variance,
                    effort=res.effort,
                    certificate=cert,
                )
            )
    return ChsFront(
        effort_cap=float(effort_cap),
        points=_mark_chs_dominance(points),
        hypotheses=tuple(dict.fromkeys(hyps)),
    )


def _mark_chs_dominance(points: list[ChsPoint]) -> tuple[ChsPoint, ...]:
    wrapped = [
        ParetoPoint(
            eps=0.0,
            status=SolveStatus.OPTIMAL
            if p.variance is not None
            else SolveStatus.NUMERICAL,
            primary=p.variance,
            constrained=-p.chs_mps,
        )
        for p in points
    ]
    marked = _mark_dominated(wrapped)
    return tuple(
        replace(p, dominated=m.dominated) for p, m in zip(points, marked, strict=True)
    )
