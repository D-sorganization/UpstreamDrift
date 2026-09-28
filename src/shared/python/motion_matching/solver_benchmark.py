"""Head-to-head scoring of matching-pipeline trajectory solvers (#11058).

Every solver is scored on the same number: replay marker RMS of its reference
through the shared MuJoCo ``FullBodySimulator``, read from the pipeline
receipt. For the baseline and the shooting fit that is ``dynamics.marker_rms_m``;
for the MJX knot optimisers it is ``trajectory_optimiser.shared_simulator_replay``,
because the MJX plant is not the scoring plant.

Rows are built only from receipts a run wrote. A solver that could not run is
an ``unavailable`` row with its reason and no numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.shared.python.core.contracts import require

G1_TARGET_M = 0.025
MJX_SOLVERS = frozenset({"mjx-adam", "mjx-lbfgs"})
PIPELINE_SOLVERS = frozenset({"none", "shooting"})
SOLVERS = PIPELINE_SOLVERS | MJX_SOLVERS | {"ipopt"}


def _downswing_min(weight_fraction: Mapping[str, Any] | None) -> float | None:
    phases = (weight_fraction or {}).get("by_phase") or {}
    downswing = phases.get("downswing")
    return None if downswing is None else float(downswing["min"])


def _worst_segment(segments: Mapping[str, float]) -> str:
    return max(segments, key=lambda name: segments[name])


def row_from_receipt(
    capture: str,
    solver: str,
    receipt: Mapping[str, Any],
    *,
    wall_clock_s: float,
    mjx_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one pipeline run; raises ``ValueError`` if a required field is absent."""
    require(solver in PIPELINE_SOLVERS | MJX_SOLVERS, f"unknown solver {solver!r}")
    require(wall_clock_s >= 0.0, "wall_clock_s must be >= 0")
    dynamics = receipt["dynamics"]
    if solver in PIPELINE_SOLVERS:
        shooting = dynamics.get("shooting_fit") or {}
        history = shooting.get("iterations", [])
        rms = float(dynamics["marker_rms_m"])
        segments = dict(dynamics["segment_rms_m"])
        downswing = _downswing_min(dynamics.get("weight_fraction"))
        iterations: int = max(0, len(history) - 1)
        evaluations: int | None = len(history) if history else 1
        stop_reason = "fixed_passes" if solver == "shooting" else "not_applicable"
    else:
        stage = receipt["trajectory_optimiser"]
        scored = stage["shared_simulator_replay"]
        rms = float(scored["replay_marker_rms_m"])
        segments = dict(scored["replay_segment_rms_m"])
        downswing = _downswing_min(scored.get("weight_fraction"))
        iterations = int(stage["iterations"])
        evaluations = len((mjx_receipt or {}).get("history", [])) or None
        stop_reason = str(stage["stop_reason"])
    return {
        "capture": capture,
        "solver": solver,
        "status": "ok",
        "replay_marker_rms_m": rms,
        "g1_met": rms <= G1_TARGET_M,
        "worst_segment": _worst_segment(segments),
        "worst_segment_rms_m": float(segments[_worst_segment(segments)]),
        "downswing_weight_fraction_min": downswing,
        "iterations": iterations,
        "evaluations": evaluations,
        "stop_reason": stop_reason,
        "wall_clock_s": float(wall_clock_s),
    }


def unavailable_row(capture: str, solver: str, reason: str) -> dict[str, Any]:
    """A solver that did not run on this host: a reason, never a number."""
    require(solver in SOLVERS, f"unknown solver {solver!r}")
    require(bool(reason), "an unavailable row needs a reason")
    return {
        "capture": capture,
        "solver": solver,
        "status": "unavailable",
        "reason": reason,
    }


def promotion_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str = "mjx-adam",
    incumbent: str = "shooting",
) -> dict[str, Any]:
    """Promote ``candidate`` only with parity on every capture.

    Parity per capture: both rows ran, the candidate's replay marker RMS is no
    worse than the incumbent's, it stopped because it converged (not on the
    iteration budget or a non-finite value), and its downswing weight fraction
    stays above zero (it did not win by leaving the ground). Any failed check
    keeps the current default.
    """
    by_key = {(r["capture"], r["solver"]): r for r in rows}
    captures = sorted({r["capture"] for r in rows})
    require(bool(captures), "no benchmark rows")
    reasons: list[str] = []
    for capture in captures:
        cand, inc = by_key.get((capture, candidate)), by_key.get((capture, incumbent))
        if not cand or cand["status"] != "ok" or not inc or inc["status"] != "ok":
            reasons.append(f"{capture}: {candidate} or {incumbent} did not run")
            continue
        if cand["replay_marker_rms_m"] > inc["replay_marker_rms_m"]:
            reasons.append(
                f"{capture}: {candidate} {cand['replay_marker_rms_m'] * 1e3:.1f} mm "
                f"> {incumbent} {inc['replay_marker_rms_m'] * 1e3:.1f} mm"
            )
        if cand["stop_reason"] != "converged":
            reasons.append(
                f"{capture}: {candidate} stopped on {cand['stop_reason']}, "
                "not converged within the budget"
            )
        wf = cand["downswing_weight_fraction_min"]
        if wf is None or wf <= 0.0:
            reasons.append(
                f"{capture}: {candidate} loses ground contact in the downswing"
            )
    return {
        "candidate": candidate,
        "incumbent": incumbent,
        "promote": not reasons,
        "reasons": reasons,
    }


def _mm(value: float | None) -> str:
    return "-" if value is None else f"{value * 1e3:.1f}"


def render_report(
    rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> str:
    """Markdown report built only from ``rows``, ``decisions`` and ``provenance``."""
    lines = [
        "# MJX Knot Optimiser Head-to-Head Benchmark (#11058)",
        "",
        "Generated by `scripts/benchmark_mjx_knot_solvers.py` from the receipts in "
        "this directory; every number below comes from one of them.",
        "",
        "## Provenance",
        "",
    ]
    lines += [
        f"- {key}: `{value}`" for key, value in provenance.items() if key != "notes"
    ]
    if provenance.get("notes"):
        lines += ["", "## Solver Choices", ""]
        lines += [f"- {note}" for note in provenance["notes"]]
    lines += [
        "",
        "## Results",
        "",
        "Replay marker RMS is scored through the shared MuJoCo simulator for every "
        f"solver. G1 target: {G1_TARGET_M * 1e3:.0f} mm.",
        "",
        "| Capture | Solver | Status | Replay RMS (mm) | G1 | Worst segment (mm) "
        "| Downswing weight fraction min | Iterations | Evaluations | Stop | Wall clock (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["status"] != "ok":
            lines.append(
                f"| {r['capture']} | {r['solver']} | unavailable: {r['reason']} "
                "| - | - | - | - | - | - | - | - |"
            )
            continue
        wf = r["downswing_weight_fraction_min"]
        lines.append(
            f"| {r['capture']} | {r['solver']} | ok | {_mm(r['replay_marker_rms_m'])} "
            f"| {'yes' if r['g1_met'] else 'no'} "
            f"| {r['worst_segment']} {_mm(r['worst_segment_rms_m'])} "
            f"| {'-' if wf is None else f'{wf:.2f}'} | {r['iterations']} "
            f"| {r['evaluations'] if r['evaluations'] is not None else '-'} "
            f"| {r['stop_reason']} | {r['wall_clock_s']:.0f} |"
        )
    lines += ["", "## Promotion Decision"]
    for decision in decisions:
        verdict = "promote" if decision["promote"] else "keep the current default"
        lines += [
            "",
            f"Candidate `{decision['candidate']}` against incumbent "
            f"`{decision['incumbent']}`: **{verdict}**.",
            "",
        ]
        lines += [f"- {reason}" for reason in decision["reasons"]] or [
            "- Parity held on every capture."
        ]
    return "\n".join(lines) + "\n"
