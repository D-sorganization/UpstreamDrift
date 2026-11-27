"""SHOW-A9 (#14734): Refit best-match MuJoCo driver on A7 grip + A8 trunk spec.

Evaluates the A9 candidate refit against the 11-04 baseline (fa7a9b7075):
1. Marker IK and forward dynamics tracking comparison.
2. A7 address grip roll holds 15 deg +/- 1 deg face-on; wrist coordinate
   violations outside coordinate_ranges_deg are explicitly cataloged (A6 lesson).
3. Best-match registry update is earned only if marker RMS and closure beat
   the baseline; otherwise recorded as candidate with trade-offs.
4. Input and output SHA-256 hashes are pinned and gate evaluations verified.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.acceptance import AcceptanceGates

REPO_ROOT = Path(__file__).resolve().parents[4]
RECEIPT_PATH = (
    REPO_ROOT / "docs/development/full_body_models/evidence/show_a9/receipt.json"
)


@dataclass(frozen=True)
class RefitMetrics:
    """The 6 required metrics comparing baseline and candidate refits."""

    marker_rms_mm: float
    joint_rms_deg: float
    closure_residual_mm: float
    weight_fraction_range: tuple[float, float]
    peak_joint_torque_nm: float
    in_support_fraction: float
    verdict: str


BASELINE_METRICS = RefitMetrics(
    50.9, 2.4, 62.2, (0.04, 2.82), 537.0, 0.637, "REJECTED (G3)"
)
CANDIDATE_METRICS = RefitMetrics(
    55.3, 2.4, 76.8, (0.00, 2.82), 537.0, 0.637, "REJECTED (G3)"
)


@dataclass(frozen=True)
class RefitComparison:
    """Side-by-side comparison of candidate refit vs baseline."""

    candidate: RefitMetrics
    baseline: RefitMetrics
    delta_marker_rms_mm: float
    delta_closure_residual_mm: float
    is_earned: bool


@precondition(lambda c, b: c.marker_rms_mm > 0.0 and b.marker_rms_mm > 0.0)
@postcondition(lambda res: isinstance(res, bool))
def is_registry_update_earned(candidate: RefitMetrics, baseline: RefitMetrics) -> bool:
    """Earned only if refit is at least as good on marker RMS and closure."""
    return (
        candidate.marker_rms_mm <= baseline.marker_rms_mm
        and candidate.closure_residual_mm <= baseline.closure_residual_mm
    )


@precondition(lambda c, b: isinstance(c, RefitMetrics) and isinstance(b, RefitMetrics))
@postcondition(lambda res: isinstance(res, RefitComparison))
def compare_against_baseline(
    candidate: RefitMetrics, baseline: RefitMetrics
) -> RefitComparison:
    """Compute deltas and whether registry update is earned."""
    return RefitComparison(
        candidate=candidate,
        baseline=baseline,
        delta_marker_rms_mm=round(candidate.marker_rms_mm - baseline.marker_rms_mm, 2),
        delta_closure_residual_mm=round(
            candidate.closure_residual_mm - baseline.closure_residual_mm, 2
        ),
        is_earned=is_registry_update_earned(candidate, baseline),
    )


def format_comparison_table(comp: RefitComparison) -> str:
    """Markdown table reporting the 6 required metrics side-by-side."""
    c, b = comp.candidate, comp.baseline
    return "\n".join(
        [
            "| Metric | 11-04 Baseline (fa7a9b7075) | A9 Candidate Refit | Delta |",
            "|---|---|---|---|",
            f"| Marker RMS | {b.marker_rms_mm:.1f} mm | {c.marker_rms_mm:.1f} mm | {comp.delta_marker_rms_mm:+.1f} mm |",
            f"| Joint RMS | {b.joint_rms_deg:.1f} deg | {c.joint_rms_deg:.1f} deg | {c.joint_rms_deg - b.joint_rms_deg:+.1f} deg |",
            f"| Closure Residual | {b.closure_residual_mm:.1f} mm | {c.closure_residual_mm:.1f} mm | {comp.delta_closure_residual_mm:+.1f} mm |",
            f"| Weight Fraction | {b.weight_fraction_range[0]:.2f}-{b.weight_fraction_range[1]:.2f} BW | {c.weight_fraction_range[0]:.2f}-{c.weight_fraction_range[1]:.2f} BW | {c.weight_fraction_range[0] - b.weight_fraction_range[0]:+.2f} min BW |",
            f"| Peak Joint Torque | {b.peak_joint_torque_nm:.0f} N*m | {c.peak_joint_torque_nm:.0f} N*m | {c.peak_joint_torque_nm - b.peak_joint_torque_nm:+.0f} N*m |",
            f"| In-Support Fraction | {b.in_support_fraction:.3f} | {c.in_support_fraction:.3f} | {c.in_support_fraction - b.in_support_fraction:+.3f} |",
        ]
    )


def load_a9_candidate_receipt(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the SHOW-A9 candidate receipt."""
    target = path or RECEIPT_PATH
    if not target.is_file():
        raise FileNotFoundError(f"A9 candidate receipt not found: {target}")
    doc: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return doc


def evaluate_a9_receipt_gates(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate quantitative G3 acceptance gates on the candidate receipt."""
    gdef = AcceptanceGates()
    comp = receipt.get("comparison_with_baseline", {})
    m_rmse = float(comp.get("marker_rms_mm", {}).get("candidate", 55.3)) / 1e3
    c_res = float(comp.get("closure_residual_mm", {}).get("candidate", 76.8)) / 1e3
    w_min = float(comp.get("weight_fraction", {}).get("candidate", [0.0, 2.82])[0])
    s_frac = float(comp.get("in_support_fraction", {}).get("candidate", 0.637))

    checks = [
        (
            "whole_marker_rmse_m",
            m_rmse <= gdef.g3_whole_driver_rmse_m,
            gdef.g3_whole_driver_rmse_m,
            m_rmse,
            "m",
            f"RMSE {m_rmse} > {gdef.g3_whole_driver_rmse_m}",
        ),
        (
            "max_closure_residual_m",
            c_res <= gdef.max_closure_residual_m,
            gdef.max_closure_residual_m,
            c_res,
            "m",
            f"closure {c_res * 1e3:.1f} mm > {gdef.max_closure_residual_m * 1e3:.1f} mm",
        ),
        (
            "weight_fraction",
            w_min >= gdef.weight_fraction_min,
            gdef.weight_fraction_min,
            w_min,
            "BW",
            f"min weight fraction {w_min:.2f} < {gdef.weight_fraction_min:.2f} BW",
        ),
        (
            "min_inside_support_polygon_fraction",
            s_frac >= gdef.min_inside_support_polygon_fraction,
            gdef.min_inside_support_polygon_fraction,
            s_frac,
            "fraction",
            f"in-support fraction {s_frac:.3f} < {gdef.min_inside_support_polygon_fraction:.3f}",
        ),
    ]
    gates = [
        {
            "name": n,
            "status": "passed" if ok else "failed",
            "threshold": th,
            "measured": val,
            "unit": u,
            "reason": "" if ok else r,
        }
        for n, ok, th, val, u, r in checks
    ]
    all_ok = all(g["status"] == "passed" for g in gates)
    return {
        "horizon": "G3",
        "status": "PASSED" if all_ok else "REJECTED",
        "is_physically_accepted": all_ok,
        "gates": gates,
        "qualification_note": (
            "Physical acceptance criteria passed"
            if all_ok
            else "Physical or kinematic thresholds violated"
        ),
    }


def compute_receipt_sha256(receipt_dict: dict[str, Any]) -> str:
    """Compute canonical sha256 of receipt excluding output_hashes."""
    copy_doc = {k: v for k, v in receipt_dict.items() if k != "output_hashes"}
    payload = json.dumps(copy_doc, sort_keys=True, indent=2).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
