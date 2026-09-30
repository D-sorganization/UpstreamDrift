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

import numpy as np

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
    stage_timings: dict[str, float] = {}
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
        raw_timings = (
            stage.get("stage_timings_s")
            or receipt.get("stage_timings_s")
            or (mjx_receipt or {}).get("stage_timings_s")
        )
        if raw_timings and isinstance(raw_timings, Mapping):
            stage_timings = {str(k): float(v) for k, v in raw_timings.items()}

    is_synthetic = bool(receipt.get("is_synthetic", False))
    capture_hash = str(receipt["capture_hash"]) if "capture_hash" in receipt else None
    geometry_hash = (
        str(receipt["geometry_hash"]) if "geometry_hash" in receipt else None
    )

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
        "stage_timings_s": stage_timings,
        "is_synthetic": is_synthetic,
        "capture_hash": capture_hash,
        "geometry_hash": geometry_hash,
    }


def _aggregate_stage_timings(successful: list[Mapping[str, Any]]) -> dict[str, float]:
    stage_timings: dict[str, float] = {}
    stage_keys: set[str] = set()
    for r in successful:
        if r.get("stage_timings_s") and isinstance(r["stage_timings_s"], Mapping):
            stage_keys.update(r["stage_timings_s"].keys())
    for sk in sorted(stage_keys):
        s_vals = [
            float(r["stage_timings_s"][sk])
            for r in successful
            if r.get("stage_timings_s") and sk in r["stage_timings_s"]
        ]
        if s_vals:
            stage_timings[sk] = float(np.median(s_vals))
    return stage_timings


def _aggregate_group_runs(
    capture: str, solver: str, group_runs: list[Mapping[str, Any]]
) -> dict[str, Any]:
    total_runs = len(group_runs)
    successful = [r for r in group_runs if r.get("status") == "ok"]
    failed = [r for r in group_runs if r.get("status") != "ok"]
    failed_count = len(failed)
    successful_count = len(successful)
    success_rate = successful_count / total_runs if total_runs > 0 else 0.0

    if not successful:
        reasons = [str(r.get("reason", "unknown failure")) for r in failed]
        return {
            "capture": capture,
            "solver": solver,
            "status": "failed",
            "total_runs": total_runs,
            "successful_runs": 0,
            "failed_runs": failed_count,
            "success_rate": 0.0,
            "reasons": reasons,
        }

    rms_vals = [float(r["replay_marker_rms_m"]) for r in successful]
    clock_vals = [float(r["wall_clock_s"]) for r in successful]

    med_rms = float(np.median(rms_vals))
    p95_rms = float(np.percentile(rms_vals, 95))
    med_clock = float(np.median(clock_vals))
    p95_clock = float(np.percentile(clock_vals, 95))

    wf_vals = [
        float(r["downswing_weight_fraction_min"])
        for r in successful
        if r.get("downswing_weight_fraction_min") is not None
    ]
    wf_min = min(wf_vals) if wf_vals else None

    worst_seg = successful[0].get("worst_segment", "unknown")
    worst_seg_rms = max(float(r.get("worst_segment_rms_m", 0.0)) for r in successful)

    stop_reasons = {str(r.get("stop_reason")) for r in successful}
    if "max_iterations" in stop_reasons:
        stop_reason = "max_iterations"
    elif len(stop_reasons) == 1:
        stop_reason = next(iter(stop_reasons))
    else:
        stop_reason = ",".join(sorted(stop_reasons))

    stage_timings = _aggregate_stage_timings(successful)

    is_synthetic = any(bool(r.get("is_synthetic")) for r in group_runs)
    cap_hashes = {str(r["capture_hash"]) for r in group_runs if r.get("capture_hash")}
    geo_hashes = {str(r["geometry_hash"]) for r in group_runs if r.get("geometry_hash")}
    capture_hash = next(iter(cap_hashes)) if len(cap_hashes) == 1 else None
    geometry_hash = next(iter(geo_hashes)) if len(geo_hashes) == 1 else None

    status = "ok" if failed_count == 0 else "partial_failure"

    return {
        "capture": capture,
        "solver": solver,
        "status": status,
        "replay_marker_rms_m": med_rms,
        "median_replay_marker_rms_m": med_rms,
        "p95_replay_marker_rms_m": p95_rms,
        "g1_met": bool(p95_rms <= G1_TARGET_M),
        "worst_segment": worst_seg,
        "worst_segment_rms_m": worst_seg_rms,
        "downswing_weight_fraction_min": wf_min,
        "iterations": int(np.median([r.get("iterations", 0) for r in successful])),
        "evaluations": int(
            np.median([r.get("evaluations", 1) or 1 for r in successful])
        ),
        "stop_reason": stop_reason,
        "wall_clock_s": med_clock,
        "median_wall_clock_s": med_clock,
        "p95_wall_clock_s": p95_clock,
        "total_runs": total_runs,
        "successful_runs": successful_count,
        "failed_runs": failed_count,
        "success_rate": success_rate,
        "stage_timings_s": stage_timings,
        "is_synthetic": is_synthetic,
        "capture_hash": capture_hash,
        "geometry_hash": geometry_hash,
    }


def aggregate_seeded_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated seeded benchmark runs for the same (capture, solver).

    Guarantees:
    - Failures remain in the denominator (total_runs = successful + failed).
    - Median and p95 metrics are calculated independently across runs.
    - Stage timings are aggregated across successful runs.
    """
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        key = (str(r["capture"]), str(r["solver"]))
        groups.setdefault(key, []).append(r)

    return [
        _aggregate_group_runs(capture, solver, runs)
        for (capture, solver), runs in groups.items()
    ]


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


def _check_capture_parity(
    capture: str,
    cand: Mapping[str, Any],
    inc: Mapping[str, Any],
    candidate: str,
    incumbent: str,
    allow_synthetic: bool,
) -> list[str]:
    reasons: list[str] = []
    if cand.get("status") not in ("ok", "partial_failure") or inc.get("status") not in (
        "ok",
        "partial_failure",
    ):
        return [f"{capture}: {candidate} or {incumbent} did not run successfully"]

    if not allow_synthetic and (cand.get("is_synthetic") or inc.get("is_synthetic")):
        reasons.append(
            f"{capture}: synthetic fixture cannot be promoted into production"
        )

    cand_cap, inc_cap = cand.get("capture_hash"), inc.get("capture_hash")
    if cand_cap and inc_cap and cand_cap != inc_cap:
        reasons.append(
            f"{capture}: capture hash mismatch ({cand_cap[:8]} != {inc_cap[:8]})"
        )

    cand_geo, inc_geo = cand.get("geometry_hash"), inc.get("geometry_hash")
    if cand_geo and inc_geo and cand_geo != inc_geo:
        reasons.append(
            f"{capture}: geometry hash mismatch ({cand_geo[:8]} != {inc_geo[:8]})"
        )

    if cand.get("failed_runs", 0) > 0 or cand.get("success_rate", 1.0) < 1.0:
        reasons.append(
            f"{capture}: {candidate} has {cand.get('failed_runs', 0)}/{cand.get('total_runs', 1)} failed runs "
            f"(success rate: {cand.get('success_rate', 0.0) * 100:.1f}%)"
        )

    cand_rms = (
        cand.get("p95_replay_marker_rms_m")
        if cand.get("p95_replay_marker_rms_m") is not None
        else cand.get("replay_marker_rms_m")
    )
    inc_rms = (
        inc.get("median_replay_marker_rms_m")
        if inc.get("median_replay_marker_rms_m") is not None
        else inc.get("replay_marker_rms_m")
    )
    if cand_rms is not None and inc_rms is not None and cand_rms > inc_rms:
        reasons.append(
            f"{capture}: {candidate} {cand_rms * 1e3:.1f} mm "
            f"> {incumbent} {inc_rms * 1e3:.1f} mm"
        )

    cand_stop = cand.get("stop_reason")
    if cand_stop != "converged":
        reasons.append(
            f"{capture}: {candidate} stopped on {cand_stop}, "
            "not converged within the budget"
        )

    wf = cand.get("downswing_weight_fraction_min")
    if wf is None or wf <= 0.0:
        reasons.append(f"{capture}: {candidate} loses ground contact in the downswing")

    return reasons


def promotion_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str = "mjx-adam",
    incumbent: str = "shooting",
    allow_synthetic: bool = True,
) -> dict[str, Any]:
    """Promote ``candidate`` only with parity on every capture."""
    by_key = {(r["capture"], r["solver"]): r for r in rows}
    captures = sorted({r["capture"] for r in rows})
    require(bool(captures), "no benchmark rows")
    reasons: list[str] = []
    for capture in captures:
        cand, inc = by_key.get((capture, candidate)), by_key.get((capture, incumbent))
        if not cand or not inc:
            reasons.append(f"{capture}: {candidate} or {incumbent} did not run")
            continue
        reasons.extend(
            _check_capture_parity(
                capture, cand, inc, candidate, incumbent, allow_synthetic
            )
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
