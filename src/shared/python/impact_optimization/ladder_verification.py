"""Per-speed verification and verdict of a full-body ladder (SHOW-C3c 3b, #14559).

The ladder is searched cheaply (closure projection off, few substeps; measured
within 1e-5 of the default policy on the v2 window).  This module re-plays each
rung's solved torque with the *default* policy and builds the verdict from that
run, never from the search (supervisor D24, show-review gate).

Rules encoded here:
    * D29: ROM and velocity limits are HARD on an accepted rung.  A rung that
      reaches its speed but breaches a limit is not accepted, and is not
      dropped: it is reported with the breaching joint, flagged
      "probable miscalibration" of the limit (or of the split it rests on),
      never as a measured over-rotation.
    * 125 mph stays firm: an unreached top rung is reported as not reached,
      naming the binding joint, and is never relabelled as a lower target.
    * Every verdict is worded "within the controlled subset", with the joints
      held at baseline listed.

Design by Contract:
    Preconditions: one check per rung; outputs are ``.json`` (never ``.npz``).
    Postconditions: the summary is JSON-serialisable and has one row per rung.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require
from src.shared.python.impact_optimization.full_body_ladder import LadderRun
from src.shared.python.impact_optimization.full_body_rollout import (
    RolloutWindow,
    make_full_body_rollout,
)
from src.shared.python.impact_optimization.impact_target import ImpactTarget
from src.shared.python.impact_optimization.kinematic_binding import KinematicBinding
from src.shared.python.impact_optimization.speed_continuation import (
    DEFAULT_SPEED_TOL_MPH,
    ContinuationPoint,
)
from src.shared.python.impact_optimization.trajectory_optimiser import (
    ImpactTolerance,
    impact_errors_deg,
)

SCOPE = "within the controlled subset"
_MISCAL = (
    "probable miscalibration of the limit (or of the split it rests on), "
    "not a measured over-rotation"
)


@dataclass(frozen=True)
class FullFidelityCheck:
    """One rung re-played with the window's (default) policy."""

    speed_target_mph: float
    search_speed_mph: float
    full_speed_mph: float
    face_error_deg: float
    path_error_deg: float
    aoa_error_deg: float
    tolerances_met: bool


def _check(
    point: ContinuationPoint,
    full: Any,
    target: ImpactTarget,
    tol: ImpactTolerance,
    speed_tol_mph: float,
) -> FullFidelityCheck:
    face, path, aoa = impact_errors_deg(full, target)
    speed = float(full.speed_mph)
    met = (
        not math.isnan(face)
        and not math.isnan(path)
        and abs(face) <= tol.face_deg
        and abs(path) <= tol.path_deg
        and abs(aoa) <= tol.aoa_deg
        and abs(speed - point.speed_target_mph) <= speed_tol_mph
    )
    return FullFidelityCheck(
        point.speed_target_mph, point.achieved_mph, speed, face, path, aoa, met
    )


def verify_at_full_fidelity(
    run: LadderRun,
    window_full: RolloutWindow,
    *,
    base_target: ImpactTarget | None = None,
    tolerance: ImpactTolerance | None = None,
    speed_tol_mph: float = DEFAULT_SPEED_TOL_MPH,
) -> tuple[FullFidelityCheck, ...]:
    """Re-play every rung's torque on ``window_full`` and score it."""
    rollout = make_full_body_rollout(window_full, run.controlled)
    base = base_target or ImpactTarget()
    tol = tolerance or ImpactTolerance()
    checks = []
    for point in run.result.points:
        outcome = rollout(point.torque_nm)
        target = dataclasses.replace(base, speed_mph=point.speed_target_mph)
        checks.append(_check(point, outcome.impact, target, tol, speed_tol_mph))
    return tuple(checks)


def _hit(h: KinematicBinding) -> dict[str, Any]:
    return {
        "joint": h.joint,
        "kind": h.kind,
        "peak": float(h.peak),
        "limit": float(h.limit),
        "utilisation": float(h.utilisation),
    }


def _row(
    point: ContinuationPoint,
    hits: tuple[KinematicBinding, ...],
    check: FullFidelityCheck,
) -> dict[str, Any]:
    breaches = [f"{h.joint} {h.kind}" for h in hits if h.utilisation > 1.0]
    return {
        "target_mph": point.speed_target_mph,
        "search_mph": check.search_speed_mph,
        "full_mph": check.full_speed_mph,
        "search_feasible": bool(point.feasible),
        "full_tolerances_met": check.tolerances_met,
        "accepted": bool(point.feasible and check.tolerances_met and not breaches),
        "torque_binding": list(point.solution.binding_joints),
        "kinematic": [_hit(h) for h in hits],
        "probable_miscalibration": breaches,
        "face_to_path_deg": _num(point.solution.face_to_path_deg),
        "full_face_error_deg": _num(check.face_error_deg),
        "full_path_error_deg": _num(check.path_error_deg),
        "full_aoa_error_deg": _num(check.aoa_error_deg),
    }


def _num(value: float) -> float | None:
    """JSON-safe float: NaN (unobservable) becomes ``None``."""
    return None if math.isnan(value) else float(value)


def _claim(top: dict[str, Any]) -> str:
    target = f"{top['target_mph']:.0f} mph"
    if top["accepted"]:
        return f"{target} reached {SCOPE} (full-fidelity run)."
    if top["probable_miscalibration"]:
        names = ", ".join(top["probable_miscalibration"])
        return f"{target} NOT reached {SCOPE}: {names} breaches its limit; {_MISCAL}."
    binding = ", ".join(top["torque_binding"]) or "no torque limit"
    return f"{target} NOT reached {SCOPE}: binding {binding}."


def summarise_ladder(
    run: LadderRun,
    checks: tuple[FullFidelityCheck, ...],
    limit_notes: dict[str, str],
) -> dict[str, Any]:
    """JSON-ready verdict built from the full-fidelity checks."""
    points = run.result.points
    require(len(checks) == len(points), "need one full-fidelity check per rung")
    rows = [
        _row(p, hits, c)
        for p, hits, c in zip(points, run.kinematic_by_rung, checks, strict=True)
    ]
    accepted = [r["target_mph"] for r in rows if r["accepted"]]
    bound = {k["joint"] for r in rows for k in r["kinematic"]} | {
        j for r in rows for j in r["torque_binding"]
    }
    base = run.baseline
    return {
        "scope": SCOPE,
        "controlled": list(run.controlled),
        "held": list(run.held),
        "baseline": {
            "speed_mph": float(base.speed_mph),
            "face_deg": None if base.face_deg is None else float(base.face_deg),
            "path_deg": None if base.path_deg is None else float(base.path_deg),
            "aoa_deg": float(base.aoa_deg),
        },
        "rungs": rows,
        "reached_full_mph": max(accepted, default=0.0),
        "claim_top_rung": _claim(rows[-1]),
        "limit_notes": {j: n for j, n in limit_notes.items() if j in bound},
    }


def write_summary(summary: dict[str, Any], path: Path) -> Path:
    """Write the summary as JSON; arrays (``.npz``) never go in the repo."""
    require(Path(path).suffix == ".json", "summary must be a .json file", path)
    Path(path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return Path(path)
