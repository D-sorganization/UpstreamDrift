"""Markdown report of a speed-continuation run (SHOW-C3c slice 3, #14559).

Turns a ``ContinuationResult`` plus the per-rung ROM / velocity bindings from
``kinematic_binding.kinematic_binding`` into one table: speed, achieved speed,
face-to-path, torque bindings, kinematic bindings.  A rung that did not reach
its target is labelled ``blocked (within the controlled subset)``: the optimiser
only controls the listed joints, so a block is evidence about that subset, not
about the golfer.  ``ContinuationPoint.feasible`` covers torque, tolerances and
speed only, so a rung that reached its speed but breaches a ROM or velocity
limit is reported as ``reached; exceeds ROM/velocity limit (...)``, never ``ok``.
ROM utilisation is measured from the range centre (0 mid-range, 1 at a bound,
above 1 outside), as in ``kinematic_binding``; percentages carry one decimal so
a breach never prints as 100%.  Limits are never altered here; a limit with a provenance note
(for example a prior-derived thoracic split) has that note printed under the
table so a binding limit is never read as a measured over-rotation.

Design by Contract:
    Preconditions: one list of bindings per rung of ``result``.
    Postconditions: the text has exactly one table row per rung.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.shared.python.contracts import require
from src.shared.python.impact_optimization.kinematic_binding import KinematicBinding
from src.shared.python.impact_optimization.speed_continuation import (
    ContinuationPoint,
    ContinuationResult,
)

BLOCKED = "blocked (within the controlled subset)"
_HEADER = (
    "| Target mph | Achieved mph | Face err deg | Path err deg | AoA err deg "
    "| Face-to-path deg | Deviation | Torque binding | Kinematic binding | Status |"
)


def _kinematic_text(hit: KinematicBinding) -> str:
    unit = "deg" if hit.kind == "ROM" else "deg/s"
    return (
        f"{hit.joint} {hit.kind} {hit.peak:.1f} / {hit.limit:.1f} {unit} "
        f"({hit.utilisation:.1%})"
    )


def _status(point: ContinuationPoint, hits: Sequence[KinematicBinding]) -> str:
    if not point.feasible:
        return BLOCKED
    breaches = [h for h in hits if h.utilisation > 1.0]
    if breaches:
        names = ", ".join(f"{h.joint} {h.kind}" for h in breaches)
        return f"reached; exceeds ROM/velocity limit ({names})"
    return "ok"


def _row(point: ContinuationPoint, hits: Sequence[KinematicBinding]) -> str:
    torque = ", ".join(point.solution.binding_joints) or "-"
    kinematic = "; ".join(_kinematic_text(h) for h in hits) or "-"
    status = _status(point, hits)
    sol = point.solution
    return (
        f"| {point.speed_target_mph:.0f} | {point.achieved_mph:.1f} "
        f"| {sol.face_error_deg:+.2f} | {sol.path_error_deg:+.2f} "
        f"| {sol.aoa_error_deg:+.2f} | {sol.face_to_path_deg:+.2f} "
        f"| {sol.deviation:.2f} | {torque} | {kinematic} | {status} |"
    )


def _bound_joints(
    result: ContinuationResult, kinematic_by_rung: Sequence[Sequence[KinematicBinding]]
) -> list[str]:
    names: list[str] = []
    for point, hits in zip(result.points, kinematic_by_rung, strict=True):
        for name in (*point.solution.binding_joints, *(h.joint for h in hits)):
            if name not in names:
                names.append(name)
    return names


def render_continuation_report(
    result: ContinuationResult,
    kinematic_by_rung: Sequence[Sequence[KinematicBinding]],
    limit_notes: Mapping[str, str],
    held_joints: Sequence[str] = (),
    fidelity_note: str = "",
) -> str:
    """Markdown table of every rung with its bindings and limit provenance."""
    require(
        len(kinematic_by_rung) == len(result.points),
        "need one kinematic-binding list per rung",
        (len(kinematic_by_rung), len(result.points)),
    )
    rows = [
        _row(p, hits) for p, hits in zip(result.points, kinematic_by_rung, strict=True)
    ]
    lines = [_HEADER, "|" + " --- |" * 10, *rows, ""]
    lines.append(f"Held joints: {', '.join(held_joints) or 'none'}")
    notes = [
        f"- {name}: {limit_notes[name]}"
        for name in _bound_joints(result, kinematic_by_rung)
        if name in limit_notes
    ]
    if notes:
        lines += ["", "Limit provenance:", *notes]
    if fidelity_note:
        lines += ["", f"Fidelity: {fidelity_note}"]
    return "\n".join(lines) + "\n"
