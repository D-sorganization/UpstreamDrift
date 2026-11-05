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
    within ``speed_tol_mph`` (0.0 when none did).
"""

from __future__ import annotations

import dataclasses
import math
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


@dataclass(frozen=True)
class ContinuationResult:
    points: tuple[ContinuationPoint, ...]
    reached_mph: float


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
    reached = max((p.speed_target_mph for p in points if p.feasible), default=0.0)
    ensure(len(points) == n_steps + 1, "one point per ladder speed")
    return ContinuationResult(tuple(points), reached)


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
) -> list[str]:
    """One evidence line per binding joint per rung (blocking rungs flagged).

    TORQUE-ONLY: every line is a torque limit and says so. ROM and velocity
    binding (the known 125 mph limiters, e.g. thoracic axial and NeckInputZ)
    are NOT reported here; they need the rollout joint angles and rates and
    are reported by ``kinematic_binding.full_binding_report`` (SHOW-C3c).

    ``limit_notes`` maps a joint name to the provenance of its limit (for
    example "prior-derived estimate, not a measurement"), appended to that joint's
    lines so a binding limit is never read as a measured over-rotation.
    """
    notes = limit_notes or {}
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
            if p.feasible:
                line = head + "; at the limit but feasible"
            else:
                line = (
                    head + "; BLOCKS this speed. 125 mph is firm, so treat the "
                    "limit as probably miscalibrated, not as a cap"
                )
            if name in notes:
                line += f" (limit provenance: {notes[name]})"
            lines.append(line)
    return lines
