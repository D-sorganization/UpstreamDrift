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

Design by Contract:
    Preconditions: ``trajectory`` is ``(len(times_s), n_joints)``; ``times_s``
    strictly increasing with >= 2 samples.
    Postconditions: utilisation >= ``threshold`` for every returned entry;
    infinite (undeclared) ROM bounds never bind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.speed_continuation import (
    ContinuationResult,
    binding_report,
)

DEFAULT_THRESHOLD = 0.98


@dataclass(frozen=True)
class KinematicBinding:
    """One joint at a ROM or velocity limit (deg or deg/s)."""

    joint: str
    kind: str  # "ROM" or "velocity"
    peak: float
    limit: float
    utilisation: float


def _rom_hit(name: str, a: np.ndarray, lo: float, hi: float) -> KinematicBinding | None:
    sides = []
    if math.isfinite(hi) and hi > 0.0:
        sides.append((float(a.max()) / hi, float(a.max()), hi))
    if math.isfinite(lo) and lo < 0.0:
        sides.append((float(a.min()) / lo, float(a.min()), lo))
    if not sides:
        return None
    util, peak, limit = max(sides)
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


def _line(speed: float, hit: KinematicBinding, feasible: bool, note: str | None) -> str:
    unit = "deg" if hit.kind == "ROM" else "deg/s"
    head = (
        f"{speed:.0f} mph: {hit.joint} reaches {hit.peak:+.1f} {unit} against its "
        f"{hit.limit:+.1f} {unit} {hit.kind} limit ({hit.utilisation:.0%})"
    )
    if hit.utilisation <= 1.0 and feasible:
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
