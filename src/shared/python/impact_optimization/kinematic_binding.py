"""ROM and velocity binding from rollout joint angles (SHOW-C3c, #14559/#14561).

``speed_continuation.binding_report`` names only *torque* limits.  At 125 mph
the known limiters are range of motion (thoracic axial, NeckInputZ), which
need the rollout joint angles and rates.  This module reads them from
``RolloutOutcome.trajectory`` (columns follow ``envelope.names``) and reports a
binding joint with its limit KIND (ROM or velocity), the peak, the limit and the
utilisation.  Limits are never raised or relaxed here: a binding limit is named
with evidence, and since 125 mph is firm it is flagged as probably miscalibrated
(for the thoracic split that means the prior-derived split or the +-50 deg ROM,
not a measured over-rotation).

ROM and velocity are HARD constraints on an accepted rung (D29):
:func:`enforce_kinematic_limits` marks a rung infeasible when any joint exceeds
its range of motion or rate limit, so it cannot raise ``reached_mph``, and
:func:`require_kinematically_feasible` is the postcondition that no feasible
rung is over a limit.  The report names the joint, limit and value.

Design by Contract:
    Preconditions: joint angles are unwrapped hinge angles (no 2*pi jumps);
    ``trajectory`` is ``(len(times_s), n_joints)``; ``times_s``
    strictly increasing with >= 2 samples.
    Postconditions: utilisation >= ``threshold`` for every returned entry;
    infinite (undeclared) ROM bounds never bind.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.speed_continuation import (
    ContinuationResult,
    binding_report,
)

DEFAULT_THRESHOLD = 0.98
LIMIT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class KinematicBinding:
    """One joint at a ROM or velocity limit (deg or deg/s)."""

    joint: str
    kind: str  # "ROM" or "velocity"
    peak: float
    limit: float
    utilisation: float


def _rom_hit(name: str, a: np.ndarray, lo: float, hi: float) -> KinematicBinding | None:
    """Utilisation from the range centre: 0 mid-range, 1 at a bound, >1 outside.

    Works for one-sided ranges (elbow ``[0, 145]``) as well as symmetric ones;
    ranges missing a bound (undeclared, ``+-inf``) never bind.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    up = ((float(a.max()) - centre) / half, float(a.max()), hi)
    down = ((centre - float(a.min())) / half, float(a.min()), lo)
    util, peak, limit = max(up, down)
    return KinematicBinding(name, "ROM", peak, limit, util)


def kinematic_binding(
    trajectory: np.ndarray,
    times_s: np.ndarray,
    envelope: PhysiologicalEnvelope,
    *,
    degrees: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[KinematicBinding, ...]:
    """Joints whose ROM or peak rate reaches ``threshold`` of their limit."""
    traj = np.asarray(trajectory, dtype=float)
    t = np.asarray(times_s, dtype=float)
    n = len(envelope.names)
    require(t.ndim == 1 and t.size >= 2 and bool(np.all(np.diff(t) > 0)), "bad times")
    require(traj.shape == (t.size, n), "trajectory must be (T, n_joints)", traj.shape)
    ang = traj if degrees else np.degrees(traj)
    rate = np.gradient(ang, t, axis=0)
    hits: list[KinematicBinding] = []
    for i, name in enumerate(envelope.names):
        lo, hi = (float(v) for v in envelope.rom_deg[i])
        rom = _rom_hit(name, ang[:, i], lo, hi)
        peak_rate = float(rate[np.argmax(np.abs(rate[:, i])), i])
        limit = float(envelope.vel_max_deg_s[i])
        vel = KinematicBinding(
            name, "velocity", peak_rate, limit, abs(peak_rate) / limit
        )
        hits.extend(
            h for h in (rom, vel) if h is not None and h.utilisation >= threshold
        )
    return tuple(hits)


def over_limit(hits: tuple[KinematicBinding, ...]) -> bool:
    """True when any binding is beyond its limit (utilisation > 1)."""
    return any(h.utilisation > 1.0 + LIMIT_TOLERANCE for h in hits)


def require_kinematically_feasible(
    result: ContinuationResult,
    kinematic_by_rung: tuple[tuple[KinematicBinding, ...], ...],
) -> None:
    """Postcondition: no feasible rung exceeds a ROM or velocity limit."""
    require(
        len(kinematic_by_rung) == len(result.points),
        "one kinematic binding tuple per rung",
    )
    bad = [
        p.speed_target_mph
        for p, hits in zip(result.points, kinematic_by_rung, strict=True)
        if p.feasible and over_limit(hits)
    ]
    ensure(not bad, "feasible rung exceeds a ROM or velocity limit", bad)


def enforce_kinematic_limits(
    result: ContinuationResult,
    envelope: PhysiologicalEnvelope,
    times_s: np.ndarray,
    *,
    degrees: bool = False,
) -> tuple[ContinuationResult, tuple[tuple[KinematicBinding, ...], ...]]:
    """Demote rungs that break a ROM or velocity limit; recompute ``reached_mph``.

    Returns the constrained result and the per-rung kinematic bindings.  Solutions
    are untouched: the over-range rung stays visible, flagged infeasible.
    """
    kin = tuple(
        kinematic_binding(p.trajectory, times_s, envelope, degrees=degrees)
        for p in result.points
    )
    points = tuple(
        dataclasses.replace(p, feasible=p.feasible and not over_limit(hits))
        for p, hits in zip(result.points, kin, strict=True)
    )
    reached = max((p.speed_target_mph for p in points if p.feasible), default=0.0)
    out = ContinuationResult(points, reached)
    require_kinematically_feasible(out, kin)
    return out, kin


def _line(speed: float, hit: KinematicBinding, feasible: bool, note: str | None) -> str:
    unit = "deg" if hit.kind == "ROM" else "deg/s"
    head = (
        f"{speed:.0f} mph: {hit.joint} reaches {hit.peak:+.1f} {unit} against its "
        f"{hit.limit:+.1f} {unit} {hit.kind} limit ({hit.utilisation:.0%})"
    )
    if hit.utilisation <= 1.0 + LIMIT_TOLERANCE and feasible:
        line = head + "; at the limit but feasible"
    else:
        line = (
            head + "; BLOCKS this speed. 125 mph is firm: probable miscalibration "
            "of the limit (or of the split it rests on), not a measured over-rotation"
        )
    return line + (f" (limit provenance: {note})" if note else "")


def kinematic_binding_report(
    result: ContinuationResult,
    envelope: PhysiologicalEnvelope,
    times_s: np.ndarray,
    *,
    degrees: bool = False,
    limit_notes: dict[str, str] | None = None,
) -> list[str]:
    """One evidence line per ROM or velocity binding per rung."""
    notes = limit_notes or {}
    lines: list[str] = []
    for p in result.points:
        traj = p.trajectory
        for hit in kinematic_binding(traj, times_s, envelope, degrees=degrees):
            lines.append(
                _line(p.speed_target_mph, hit, p.feasible, notes.get(hit.joint))
            )
    return lines


def full_binding_report(
    result: ContinuationResult,
    envelope: PhysiologicalEnvelope,
    times_s: np.ndarray,
    *,
    degrees: bool = False,
    limit_notes: dict[str, str] | None = None,
) -> list[str]:
    """Torque, ROM and velocity binding lines; every line names its limit kind."""
    return binding_report(result, envelope, limit_notes) + kinematic_binding_report(
        result, envelope, times_s, degrees=degrees, limit_notes=limit_notes
    )
