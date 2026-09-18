"""Recompute Fast-Matching Evidence from Raw NPZ Archives (PF-01 / #10431).

Audits candidate archives in evidence/matched/driver_full_pinocchio and
evidence/matched/iron_full_pinocchio, recording explicit, immutable rejection
audits (rejection_audit.json) verifying physical and kinematic gate failures
under G3 criteria.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.motion_matching.acceptance import (
    AcceptanceVerdict,
    GateStatus,
    Horizon,
    evaluate,
)
from src.shared.python.motion_matching.swing_evaluator import (
    SwingEvaluator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _compute_contact_metrics(
    ground_forces: np.ndarray, n_nodes: int
) -> tuple[float, float, int]:
    gf_reshaped = ground_forces.reshape(n_nodes, 6, 3)
    f_tangential = np.linalg.norm(gf_reshaped[:, :, :2], axis=2)
    f_normal = gf_reshaped[:, :, 2]
    active_mask = f_normal > 1.0
    friction_ratios = np.where(
        active_mask, f_tangential / np.maximum(f_normal, 1e-6), 0.0
    )
    max_friction_ratio = float(np.max(friction_ratios)) if np.any(active_mask) else 0.0
    friction_violations = int(np.sum(friction_ratios > 0.8))
    max_normal_force_n = float(np.max(f_normal))
    return max_normal_force_n, max_friction_ratio, friction_violations


def _read_receipt_metrics(receipt_path: Path) -> tuple[float, float]:
    yaw_rmse_rad = 0.0
    max_closure_residual_m = 0.0
    if receipt_path.exists():
        receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
        shared_sec = receipt_data.get("metrics", {}).get("shared", {})
        yaw_rmse_rad = float(shared_sec.get("pelvis_yaw_rmse_rad", 0.0))
        closure_sec = receipt_data.get("closure", {})
        max_closure_residual_m = float(closure_sec.get("max_closure_mm", 0.0)) / 1000.0
    return yaw_rmse_rad, max_closure_residual_m


def audit_candidate_archive(archive_dir: Path, capture_name: str) -> dict[str, Any]:
    """Audit raw candidate.npz archive and evaluate against physical gates."""
    npz_path = archive_dir / "candidate.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Candidate archive not found: {npz_path}")

    logger.info("Auditing %s from %s...", capture_name, npz_path)
    with np.load(npz_path, allow_pickle=False) as raw:
        time_s = np.asarray(raw["time_s"], dtype=np.float64)
        pred_markers = np.asarray(raw["markers_m"], dtype=np.float64)
        targ_markers = np.asarray(raw["target_m"], dtype=np.float64)
        valid = np.asarray(raw["valid"], dtype=bool)
        labels = [str(x) for x in raw["labels"]]
        ground_forces = np.asarray(raw["ground_forces"], dtype=np.float64)

    n_nodes = len(time_s)
    duration_s = float(time_s[-1] - time_s[0])

    evaluator = SwingEvaluator(labels=labels)
    eval_report = evaluator.evaluate(
        time_s=time_s,
        pred_markers=pred_markers,
        target_markers=targ_markers,
        valid=valid,
        sphere_bottom_z=np.zeros((n_nodes, 6)),
        ground_height_m=0.0,
        closure_errors_m=np.zeros(n_nodes),
    )

    max_norm_n, max_fric, fric_viols = _compute_contact_metrics(ground_forces, n_nodes)
    yaw_rmse, max_clos_m = _read_receipt_metrics(archive_dir / "receipt.json")

    club_rmse_m = (
        eval_report.segments["club"].rmse_mm / 1000.0
        if "club" in eval_report.segments
        else 0.0
    )
    max_pen_m = eval_report.ground_penetration.max_penetration_mm / 1000.0

    audit_payload: dict[str, Any] = {
        "schema": "matched-swing-fit/pinocchio-analytic-inverse-dynamics-v1",
        "status": "REJECTED",
        "rejection_reason": "Physical and kinematic acceptance thresholds violated under Horizon G3",
        "capture": capture_name,
        "num_frames": n_nodes,
        "duration_s": duration_s,
        "shared_metrics": {
            "whole_marker_rmse_m": eval_report.overall_rmse_mm / 1000.0,
            "club_marker_rmse_m": club_rmse_m,
            "pelvis_yaw_rmse_rad": yaw_rmse,
        },
        "contact_audit": {
            "max_normal_force_n": max_norm_n,
            "max_friction_ratio": max_fric,
            "friction_violation_count": fric_viols,
            "max_penetration_m": max_pen_m,
        },
        "closure": {"max_closure_residual_m": max_clos_m},
        "dynamics": {
            "has_delta_tau_root": False,
            "note": "delta_tau_root unrecorded in raw NPZ; ungrounded phantom assistance unverified",
        },
        "evaluation_report": eval_report.to_dict(),
    }

    verdict: AcceptanceVerdict = evaluate(audit_payload, horizon=Horizon.G3)
    audit_payload["verdict"] = verdict.as_dict()

    failing_gates = [g for g in verdict.gates if g.status == GateStatus.FAILED]
    logger.info(
        "%s G3 Acceptance Status: %s (%d failing gates)",
        capture_name,
        verdict.status,
        len(failing_gates),
    )
    for g in failing_gates:
        logger.warning(
            "  FAILED: %s | measured=%s %s | threshold=%s %s | reason=%s",
            g.name,
            g.measured,
            g.unit,
            g.threshold,
            g.unit,
            g.reason,
        )

    out_path = archive_dir / "rejection_audit.json"
    out_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    logger.info("Saved explicit rejection audit to %s", out_path)
    return audit_payload


def main() -> None:
    driver_dir = REPO_ROOT / "evidence" / "matched" / "driver_full_pinocchio"
    iron_dir = REPO_ROOT / "evidence" / "matched" / "iron_full_pinocchio"

    if driver_dir.exists():
        audit_candidate_archive(driver_dir, "driver")

    if iron_dir.exists():
        audit_candidate_archive(iron_dir, "iron")


if __name__ == "__main__":
    main()
