"""Report and diagnostics helpers for OpenSim barbell lift joint reactions.

Provides structured records, phase summarization, static-hold analytic checks,
and Movement Optimizer comparison for lift joint reactions (OSL-5, #13043).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from src.shared.python.contracts import require


@dataclass(frozen=True)
class JointReactionSeries:
    """Frame-by-frame joint reactions across a lift motion window."""

    time_s: np.ndarray
    l5s1_comp_n: np.ndarray
    l5s1_ap_shear_n: np.ndarray
    l5s1_ml_shear_n: np.ndarray
    l5s1_resultant_n: np.ndarray
    knee_tf_r_n: np.ndarray
    knee_tf_l_n: np.ndarray
    knee_pf_r_n: np.ndarray
    knee_pf_l_n: np.ndarray
    shoulder_gh_r_n: np.ndarray
    shoulder_gh_l_n: np.ndarray
    max_acc_error: float


def unavailable_joint_reaction_record(lift: str, reason: str) -> dict[str, Any]:
    """Return a standard record for lifts lacking an OSL-4 muscle solve or runtime."""
    return {
        "lift": lift,
        "status": "unavailable",
        "reason": reason,
        "classification": "unqualified",
        "window_start_s": None,
        "window_end_s": None,
        "duration_s": None,
        "body_weight_n": None,
        "acceleration_error_max": None,
        "l5_s1": None,
        "knee_tibiofemoral": None,
        "knee_patellofemoral": None,
        "shoulder_glenohumeral": None,
        "cross_check_movement_optimizer": None,
        "static_hold_analytic_bound": None,
        "phases": None,
    }


def summarize_phase_reactions(
    phase_segments: Sequence[tuple[str, float, float]],
    times: np.ndarray,
    reactions: JointReactionSeries,
    body_weight_n: float,
) -> dict[str, dict[str, float]]:
    """Summarize mean and peak reaction loads per motion phase."""
    require(len(times) > 0, "times must not be empty")
    require(body_weight_n > 0.0, "body_weight_n must be positive")

    summary: dict[str, dict[str, float]] = {}
    for label, t0, t1 in phase_segments:
        mask = (times >= t0 - 1e-9) & (times <= t1 + 1e-9)
        if not np.any(mask):
            continue

        c = reactions.l5s1_comp_n[mask]
        s = reactions.l5s1_ap_shear_n[mask]
        tf = 0.5 * (reactions.knee_tf_r_n[mask] + reactions.knee_tf_l_n[mask])
        pf = 0.5 * (reactions.knee_pf_r_n[mask] + reactions.knee_pf_l_n[mask])
        gh = 0.5 * (reactions.shoulder_gh_r_n[mask] + reactions.shoulder_gh_l_n[mask])

        summary[label] = {
            "l5_s1_compression_mean_n": round(float(np.mean(c)), 2),
            "l5_s1_compression_peak_n": round(float(np.max(c)), 2),
            "l5_s1_compression_mean_bw": round(float(np.mean(c) / body_weight_n), 2),
            "l5_s1_compression_peak_bw": round(float(np.max(c) / body_weight_n), 2),
            "l5_s1_ap_shear_mean_n": round(float(np.mean(s)), 2),
            "l5_s1_ap_shear_peak_n": round(float(s[np.argmax(np.abs(s))]), 2),
            "knee_tf_resultant_mean_n": round(float(np.mean(tf)), 2),
            "knee_tf_resultant_peak_n": round(float(np.max(tf)), 2),
            "knee_tf_resultant_mean_bw": round(float(np.mean(tf) / body_weight_n), 2),
            "knee_tf_resultant_peak_bw": round(float(np.max(tf) / body_weight_n), 2),
            "knee_pf_resultant_mean_n": round(float(np.mean(pf)), 2),
            "knee_pf_resultant_peak_n": round(float(np.max(pf)), 2),
            "knee_pf_resultant_mean_bw": round(float(np.mean(pf) / body_weight_n), 2),
            "knee_pf_resultant_peak_bw": round(float(np.max(pf) / body_weight_n), 2),
            "shoulder_gh_resultant_mean_n": round(float(np.mean(gh)), 2),
            "shoulder_gh_resultant_peak_n": round(float(np.max(gh)), 2),
            "shoulder_gh_resultant_mean_bw": round(
                float(np.mean(gh) / body_weight_n), 2
            ),
            "shoulder_gh_resultant_peak_bw": round(
                float(np.max(gh) / body_weight_n), 2
            ),
        }
    return summary


def format_joint_reaction_solution(
    lift: str,
    times: np.ndarray,
    body_weight_n: float,
    reactions: JointReactionSeries,
    phases: dict[str, dict[str, float]],
    cross_check: dict[str, Any],
    static_bound: dict[str, Any],
) -> dict[str, Any]:
    """Format full JSON-serializable joint reaction receipt."""
    bw = body_weight_n
    t0, t1 = float(times[0]), float(times[-1])

    c = reactions.l5s1_comp_n
    s = reactions.l5s1_ap_shear_n
    tf_r, tf_l = reactions.knee_tf_r_n, reactions.knee_tf_l_n
    pf_r, pf_l = reactions.knee_pf_r_n, reactions.knee_pf_l_n
    gh_r, gh_l = reactions.shoulder_gh_r_n, reactions.shoulder_gh_l_n

    return {
        "lift": lift,
        "status": "complete",
        "classification": "unqualified",
        "window_start_s": round(t0, 4),
        "window_end_s": round(t1, 4),
        "duration_s": round(t1 - t0, 4),
        "body_weight_n": round(bw, 2),
        "acceleration_error_max": float(reactions.max_acc_error),
        "l5_s1": {
            "joint": "back",
            "frame": "inferior vertebra ISB frame (pelvis, X anterior, Y cranial, Z right)",
            "sign_convention": "compression positive (+F_y), anterior shear positive (+F_x)",
            "compression_mean_n": round(float(np.mean(c)), 2),
            "compression_peak_n": round(float(np.max(c)), 2),
            "compression_mean_bw": round(float(np.mean(c) / bw), 2),
            "compression_peak_bw": round(float(np.max(c) / bw), 2),
            "ap_shear_mean_n": round(float(np.mean(s)), 2),
            "ap_shear_peak_n": round(float(s[np.argmax(np.abs(s))]), 2),
            "ap_shear_mean_bw": round(float(np.mean(s) / bw), 2),
            "ap_shear_peak_bw": round(float(s[np.argmax(np.abs(s))] / bw), 2),
            "resultant_mean_n": round(float(np.mean(reactions.l5s1_resultant_n)), 2),
            "resultant_peak_n": round(float(np.max(reactions.l5s1_resultant_n)), 2),
        },
        "knee_tibiofemoral": {
            "joint_r": "walker_knee_r",
            "joint_l": "walker_knee_l",
            "frame": "resultant 3D spatial force norm",
            "resultant_mean_n": round(float(np.mean(0.5 * (tf_r + tf_l))), 2),
            "resultant_peak_n": round(float(max(np.max(tf_r), np.max(tf_l))), 2),
            "resultant_mean_bw": round(float(np.mean(0.5 * (tf_r + tf_l)) / bw), 2),
            "resultant_peak_bw": round(float(max(np.max(tf_r), np.max(tf_l)) / bw), 2),
            "right_mean_n": round(float(np.mean(tf_r)), 2),
            "right_peak_n": round(float(np.max(tf_r)), 2),
            "left_mean_n": round(float(np.mean(tf_l)), 2),
            "left_peak_n": round(float(np.max(tf_l)), 2),
        },
        "knee_patellofemoral": {
            "model_has_patella_joint": True,
            "joint_r": "patellofemoral_r",
            "joint_l": "patellofemoral_l",
            "frame": "resultant 3D spatial force norm",
            "resultant_mean_n": round(float(np.mean(0.5 * (pf_r + pf_l))), 2),
            "resultant_peak_n": round(float(max(np.max(pf_r), np.max(pf_l))), 2),
            "resultant_mean_bw": round(float(np.mean(0.5 * (pf_r + pf_l)) / bw), 2),
            "resultant_peak_bw": round(float(max(np.max(pf_r), np.max(pf_l)) / bw), 2),
            "right_mean_n": round(float(np.mean(pf_r)), 2),
            "right_peak_n": round(float(np.max(pf_r)), 2),
            "left_mean_n": round(float(np.mean(pf_l)), 2),
            "left_peak_n": round(float(np.max(pf_l)), 2),
        },
        "shoulder_glenohumeral": {
            "status": "complete",
            "is_articulated": True,
            "joint_r": "acromial_r",
            "joint_l": "acromial_l",
            "frame": "resultant 3D spatial force norm",
            "resultant_mean_n": round(float(np.mean(0.5 * (gh_r + gh_l))), 2),
            "resultant_peak_n": round(float(max(np.max(gh_r), np.max(gh_l))), 2),
            "resultant_mean_bw": round(float(np.mean(0.5 * (gh_r + gh_l)) / bw), 2),
            "resultant_peak_bw": round(float(max(np.max(gh_r), np.max(gh_l)) / bw), 2),
            "right_mean_n": round(float(np.mean(gh_r)), 2),
            "right_peak_n": round(float(np.max(gh_r)), 2),
            "left_mean_n": round(float(np.mean(gh_l)), 2),
            "left_peak_n": round(float(np.max(gh_l)), 2),
            "note": "acromial_r/l is articulated CustomJoint (3 rotational DOFs, not WeldJoint)",
        },
        "cross_check_movement_optimizer": cross_check,
        "static_hold_analytic_bound": static_bound,
        "phases": phases,
        "timeseries": {
            "time_s": [round(float(v), 4) for v in times],
            "l5s1_comp_n": [round(float(v), 2) for v in c],
            "l5s1_ap_shear_n": [round(float(v), 2) for v in s],
            "knee_tf_r_n": [round(float(v), 2) for v in tf_r],
            "knee_tf_l_n": [round(float(v), 2) for v in tf_l],
            "knee_pf_r_n": [round(float(v), 2) for v in pf_r],
            "knee_pf_l_n": [round(float(v), 2) for v in pf_l],
            "shoulder_gh_r_n": [round(float(v), 2) for v in gh_r],
            "shoulder_gh_l_n": [round(float(v), 2) for v in gh_l],
        },
    }


def crosscheck_movement_optimizer_l5s1(
    targets: Any, torso_angles: np.ndarray
) -> dict[str, Any]:
    """Compute Movement Optimizer spine_loads.py L5/S1 on squat kinematics."""
    tools_src = Path(__file__).resolve().parents[7] / "vendor" / "ud-tools" / "src"
    if str(tools_src) not in sys.path:
        sys.path.insert(0, str(tools_src))

    from movement_optimizer.models import BodyModel
    from movement_optimizer.spine_loads import (
        MODEL_METADATA,
        spinal_compression,
        spinal_shear,
        to_adr0055_ap_shear,
    )

    q_opt = np.zeros((len(targets.time_s), 3))
    q_opt[:, 2] = torso_angles
    times = targets.time_s
    qd_opt = np.gradient(q_opt, times, axis=0)
    qdd_opt = np.gradient(qd_opt, times, axis=0)

    body = BodyModel(body_mass=targets.body_mass_kg, height=targets.height_m)
    bar_mass = targets.bar_mass_kg

    opt_comp = spinal_compression(q_opt, qd_opt, qdd_opt, body, bar_mass, "squat")
    opt_shear = spinal_shear(q_opt, qd_opt, qdd_opt, body, bar_mass, "squat")
    opt_adr_shear = to_adr0055_ap_shear(opt_shear)

    return {
        "model": str(MODEL_METADATA["model"]),
        "is_joint_reaction": False,
        "torso_angle_mean_deg": round(float(np.degrees(np.mean(torso_angles))), 2),
        "torso_angle_peak_deg": round(float(np.degrees(np.max(torso_angles))), 2),
        "compression_mean_n": round(float(np.mean(opt_comp)), 2),
        "compression_peak_n": round(float(np.max(opt_comp)), 2),
        "shear_mean_n": round(float(np.mean(opt_shear)), 2),
        "shear_peak_n": round(float(np.max(opt_shear)), 2),
        "adr0055_ap_shear_mean_n": round(float(np.mean(opt_adr_shear)), 2),
        "adr0055_ap_shear_peak_n": round(float(np.min(opt_adr_shear)), 2),
        "compression_curve_n": [round(float(v), 2) for v in opt_comp],
        "adr0055_ap_shear_curve_n": [round(float(v), 2) for v in opt_adr_shear],
        "model_difference_explanation": (
            "Movement Optimizer computes a planar ergonomic rigid-body estimate of skeletal "
            "load based on gravitational and inertial torso segment loads without active muscle "
            "tensions. OpenSim computes the 3D musculoskeletal joint reaction with 3D pelvic and "
            "lumbar coordinates, forward-dynamics accelerations, and contact forces."
        ),
    }
