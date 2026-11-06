"""Speed continuation 100 -> 125 mph with warm starts (SHOW-C5, #14561).

Raises the target clubhead/ball speed in small steps, warm-starting each
solve from the previous knots, and records every step's solution.  125 mph is
FIRM: a step that cannot be met is never relabelled as a cap.  When a torque
limit binds, :func:`binding_report` names the joint with the evidence and
treats the limit as probably miscalibrated, so it is surfaced, not silently
raised or silently obeyed.

Design by Contract:
    Preconditions: ``step_mph > 0`` and ``stop_mph >= start_mph``.
    Postconditions: one point per ladder speed, in order; ``reached_mph`` is
    the highest speed whose point met the angle tolerances and the speed
    within ``speed_tol_mph`` (0.0 when none did).  This function does not see
    joint angles; ``kinematic_binding.enforce_kinematic_limits`` further demotes
    rungs that break a ROM or velocity limit and recomputes ``reached_mph``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.impact_optimization.impact_target import ImpactTarget
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.trajectory_optimiser import (
    ImpactSolution,
    OptimiserConfig,
    Rollout,
    solve_impact_trajectory,
)
from src.shared.python.impact_optimization.trust_region import TrustRegion
from src.shared.python.loop_nullspace.capacity import CapacityVector

DEFAULT_SPEED_TOL_MPH: float = 1.0


@dataclass(frozen=True)
class LadderSpec:
    """Speed ladder: ``start_mph`` .. ``stop_mph`` inclusive in ``step_mph``."""

    start_mph: float = 100.0
    stop_mph: float = 125.0
    step_mph: float = 5.0
    speed_tol_mph: float = DEFAULT_SPEED_TOL_MPH

    def __post_init__(self) -> None:
        require(
            self.step_mph > 0.0 and math.isfinite(self.step_mph),
            "step_mph must be > 0",
        )
        require(self.stop_mph >= self.start_mph, "stop_mph must be >= start_mph")

    @property
    def n_steps(self) -> int:
        return int(math.floor((self.stop_mph - self.start_mph) / self.step_mph + 1e-9))


@dataclass(frozen=True)
class ContinuationPoint:
    """One rung of the ladder."""

    speed_target_mph: float
    solution: ImpactSolution
    feasible: bool

    @property
    def achieved_mph(self) -> float:
        return self.solution.achieved_speed_mph

    @property
    def trajectory(self) -> np.ndarray:
        """Rollout joint angles of this rung (delegates; keeps callers shallow)."""
        return self.solution.trajectory

    @property
    def torque_nm(self) -> np.ndarray:
        """Solved torque of this rung (delegates; keeps callers shallow)."""
        return self.solution.torque_nm

    @property
    def binding_joints(self) -> tuple[str, ...]:
        """Torque-binding joints of this rung (delegates)."""
        return self.solution.binding_joints

    @property
    def face_to_path_deg(self) -> float:
        """Face minus path of this rung (delegates)."""
        return self.solution.face_to_path_deg


@dataclass(frozen=True)
class ContinuationResult:
    points: tuple[ContinuationPoint, ...]
    reached_mph: float


def reached_mph(points: Sequence[ContinuationPoint]) -> float:
    """Highest feasible target speed (0.0 when none)."""
    return max((p.speed_target_mph for p in points if p.feasible), default=0.0)


def run_speed_continuation(
    rollout: Rollout,
    baseline_torque_nm: np.ndarray,
    times_s: np.ndarray,
    envelope: PhysiologicalEnvelope,
    trust_region: TrustRegion,
    config: OptimiserConfig | None = None,
    base_target: ImpactTarget | None = None,
    ladder: LadderSpec | None = None,
) -> ContinuationResult:
    """Solve every rung of ``ladder`` (default 100 -> 125 mph, step 5)."""
    spec = ladder or LadderSpec()
    base = base_target or ImpactTarget()
    n_steps = spec.n_steps
    points: list[ContinuationPoint] = []
    warm: np.ndarray | None = None
    for k in range(n_steps + 1):
        speed = spec.start_mph + k * spec.step_mph
        sol = solve_impact_trajectory(
            rollout,
            baseline_torque_nm,
            times_s,
            dataclasses.replace(base, speed_mph=speed),
            envelope,
            trust_region,
            config,
            initial_knots=warm,
        )
        warm = sol.knots
        ok = sol.tolerances_met and (
            abs(sol.achieved_speed_mph - speed) <= spec.speed_tol_mph
        )
        points.append(ContinuationPoint(speed, sol, ok))
    ensure(len(points) == n_steps + 1, "one point per ladder speed")
    return ContinuationResult(tuple(points), reached_mph(points))


def pareto_front(
    points: tuple[ContinuationPoint, ...],
) -> tuple[ContinuationPoint, ...]:
    """Non-dominated points: higher achieved speed, lower impact deviation."""

    def dominates(q: ContinuationPoint, p: ContinuationPoint) -> bool:
        return (
            q.achieved_mph >= p.achieved_mph
            and q.solution.deviation <= p.solution.deviation
            and (
                q.achieved_mph > p.achieved_mph
                or q.solution.deviation < p.solution.deviation
            )
        )

    front = [p for p in points if not any(dominates(q, p) for q in points)]
    return tuple(sorted(front, key=lambda p: p.achieved_mph))


def binding_report(
    result: ContinuationResult,
    envelope: PhysiologicalEnvelope,
    limit_notes: dict[str, str] | None = None,
    source_band: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> list[str]:
    """One evidence line per binding joint per rung (blocking rungs flagged).

    TORQUE-ONLY: every line is a torque limit and says so. ROM and velocity
    binding (the known 125 mph limiters, e.g. thoracic axial and NeckInputZ)
    are NOT reported here; they need the rollout joint angles and rates and
    are reported by ``kinematic_binding.full_binding_report`` (SHOW-C3c).

    ``source_band`` maps ``(joint, "positive" | "negative")`` to the (lower,
    upper) cited torque values of that direction when sources conflict (D34);
    the envelope is signed, so each side is classified on its own signed
    extreme.  The envelope limit of that side must BE the upper value, so a
    block at the ceiling is blocked under both sources and keeps the D29
    wording.  A rung whose peak is above the lower value but within the upper
    one is "source-sensitive": it holds only if the upper source applies.  This
    is distinct from "assumption-limited" (D32) and "probable miscalibration".

    ``limit_notes`` maps a joint name to the provenance of its limit (for
    example "prior-derived estimate, not a measurement"), appended to that joint's
    lines so a binding limit is never read as a measured over-rotation.
    """
    notes = limit_notes or {}
    band = source_band or {}
    _require_band(band, envelope)
    lines: list[str] = []
    for p in result.points:
        sol = p.solution
        tau = sol.outcome.torque_nm
        util = np.max(np.stack([envelope.utilisation(r) for r in tau]), axis=0)
        peak = np.max(np.abs(tau), axis=0)
        for name in sol.binding_joints:
            i = envelope.names.index(name)
            head = (
                f"{p.speed_target_mph:.0f} mph: {name} at {util[i]:.0%} of its "
                f"{sol.envelope_limits_nm[i]:.1f} N m torque limit (peak {peak[i]:.1f} N m, "
                f"achieved {p.achieved_mph:.1f} mph, face-to-path "
                f"{sol.face_to_path_deg:+.2f} deg)"
            )
            hits = _banded_sides(band, name, tau[:, i])
            if p.feasible and hits:
                line = head + "; at the limit but feasible; " + _sources_text(hits)
            elif p.feasible:
                line = head + "; at the limit but feasible"
            elif name in envelope.assumed_torque_slots:
                line = (
                    head + "; BLOCKS this speed. assumption-limited: this torque "
                    "limit is an assumed placeholder, not a cited value, so the "
                    "rung says nothing about the athlete"
                )
            else:
                line = (
                    head + "; BLOCKS this speed. 125 mph is firm, so treat the "
                    "limit as probably miscalibrated, not as a cap"
                )
            if name in notes:
                line += f" (limit provenance: {notes[name]})"
            lines.append(line)
        for name in sorted({n for n, _ in band}):
            i = envelope.names.index(name)
            hits = _banded_sides(band, name, tau[:, i])
            if name not in sol.binding_joints and hits:
                peaks = ", ".join(f"{h[3]:.1f} N m ({h[0]})" for h in hits)
                lines.append(
                    f"{p.speed_target_mph:.0f} mph: {name} peaks at {peaks}, "
                    f"within its limit; " + _sources_text(hits)
                )
    return lines


def _banded_sides(
    band: dict[tuple[str, str], tuple[float, float]], name: str, tau: np.ndarray
) -> list[tuple[str, float, float, float]]:
    """(direction, lower, upper, peak) of every side above its lower value."""
    hits = []
    for direction, peak in (
        ("positive", float(np.max(tau))),
        ("negative", float(-np.min(tau))),
    ):
        if (name, direction) in band and peak > band[(name, direction)][0]:
            low, high = band[(name, direction)]
            hits.append((direction, low, high, peak))
    return hits


def _sources_text(hits: list[tuple[str, float, float, float]]) -> str:
    return "; ".join(_source_text(*h) for h in hits)


def source_band_from_capacity(
    cap: CapacityVector, envelope: PhysiologicalEnvelope
) -> dict[tuple[str, str], tuple[float, float]]:
    """``binding_report`` bands from the registry's lower cited references (D34a).

    Only the sides with a second, lower source get a band; a side whose lower
    reference equals its capacity (single source) is skipped, so it keeps the
    D29 / D32 wording.
    """
    lo_neg = cap.tau_lower_reference_negative_nm
    lo_pos = cap.tau_lower_reference_positive_nm
    if lo_neg is None or lo_pos is None:
        require(False, "capacity has no lower references")
        return {}
    band: dict[tuple[str, str], tuple[float, float]] = {}
    for name in envelope.names:
        if name not in cap.actuated_names:
            continue
        k = cap.actuated_names.index(name)
        i = envelope.names.index(name)
        for direction, low, high in (
            ("positive", lo_pos[k], envelope.tau_max_positive_nm[i]),
            ("negative", lo_neg[k], envelope.tau_max_negative_nm[i]),
        ):
            if 0.0 < low < high:
                band[(name, direction)] = (float(low), float(high))
    return band


def _source_text(direction: str, low: float, high: float, peak: float) -> str:
    return (
        f"source-sensitive ({direction}): the peak {peak:.1f} N m is above the "
        f"lower cited value ({low:.1f} N m) and within the upper ({high:.1f} N m), "
        "so it holds only if the upper source applies"
    )


def _require_band(
    band: dict[tuple[str, str], tuple[float, float]], envelope: PhysiologicalEnvelope
) -> None:
    """Each band is ``0 < lower < upper``; its side's envelope limit is the upper."""
    for (name, direction), (low, high) in band.items():
        require(name in envelope.names, f"source_band joint not in envelope: {name}")
        require(
            direction in ("positive", "negative"),
            f"source_band direction must be positive or negative: {direction}",
        )
        require(0.0 < low < high, f"source_band {name} {direction}: 0 < lower < upper")
        side = (
            envelope.tau_max_positive_nm
            if direction == "positive"
            else envelope.tau_max_negative_nm
        )
        limit = float(side[envelope.names.index(name)])
        require(
            abs(limit - high) <= 1e-9 * max(1.0, high),
            f"source_band {name} {direction}: the envelope limit must be the upper",
            (limit, high),
        )
