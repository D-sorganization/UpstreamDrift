"""Report and diagnostics helpers for OpenSim barbell lift joint reactions (OSL-5, #13043)."""

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
    l5s1_torso_comp_n: np.ndarray
    l5s1_torso_ap_shear_n: np.ndarray
    max_acc_error: float


def unavailable_joint_reaction_record(lift: str, reason: str) -> dict[str, Any]:
    """Return a standard record for lifts lacking an OSL-4 muscle solve or runtime."""
    return {
        "lift": lift,
        "status": "unavailable",
        "reason": reason,
        "classification": "unqualified",
        "qualification": None,
        "reserve_share": None,
        "mean_reserve_share": None,
        "peak_reserve_share": None,
        "reserve_share_threshold": None,
        "window_start_s": None,
        "window_end_s": None,
        "duration_s": None,
        "body_weight_n": None,
        "lifter_mass_kg": None,
        "bar_mass_kg": None,
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

        d = {
            "l5_s1_compression_mean_n": round(float(np.mean(c)), 2),
            "l5_s1_compression_peak_n": round(float(np.max(c)), 2),
            "l5_s1_compression_mean_bw": round(float(np.mean(c) / body_weight_n), 2),
            "l5_s1_compression_peak_bw": round(float(np.max(c) / body_weight_n), 2),
            "l5_s1_ap_shear_mean_n": round(float(np.mean(s)), 2),
            "l5_s1_ap_shear_peak_n": round(float(s[np.argmax(np.abs(s))]), 2),
        }
        for prefix, arr in (
            ("knee_tf_resultant", tf),
            ("knee_pf_resultant", pf),
            ("shoulder_gh_resultant", gh),
        ):
            m, p = float(np.mean(arr)), float(np.max(arr))
            d[f"{prefix}_mean_n"] = round(m, 2)
            d[f"{prefix}_peak_n"] = round(p, 2)
            d[f"{prefix}_mean_bw"] = round(m / body_weight_n, 2)
            d[f"{prefix}_peak_bw"] = round(p / body_weight_n, 2)
        summary[label] = d
    return summary


def _format_l5s1_section(
    reactions: JointReactionSeries,
    bw: float,
    peak_lumbar_moment_nm: float | None = None,
) -> dict[str, Any]:
    c = reactions.l5s1_comp_n
    s = reactions.l5s1_ap_shear_n
    tc = reactions.l5s1_torso_comp_n
    ts = reactions.l5s1_torso_ap_shear_n
    return {
        "joint": "back",
        "frame": "inferior vertebra ISB frame (pelvis, X anterior, Y cranial, Z right)",
        "qualification": "lower bound: no trunk muscles; the lumbar moment is an ideal actuator",
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
        "compression_torso_frame_mean_n": round(float(np.mean(tc)), 2),
        "compression_torso_frame_peak_n": round(float(np.max(tc)), 2),
        "ap_shear_torso_frame_mean_n": round(float(np.mean(ts)), 2),
        "ap_shear_torso_frame_peak_n": round(float(ts[np.argmax(np.abs(ts))]), 2),
        "peak_lumbar_actuator_moment_nm": (
            round(float(peak_lumbar_moment_nm), 2)
            if peak_lumbar_moment_nm is not None
            else None
        ),
        "segmental_spine_reference": "Use segmental spine model (#12735) for multi-segment spinal load with active trunk muscles",
    }


def _bilateral_metrics(
    val_r: np.ndarray, val_l: np.ndarray, bw: float
) -> dict[str, float]:
    avg = 0.5 * (val_r + val_l)
    return {
        "resultant_mean_n": round(float(np.mean(avg)), 2),
        "resultant_peak_n": round(float(max(np.max(val_r), np.max(val_l))), 2),
        "resultant_mean_bw": round(float(np.mean(avg) / bw), 2),
        "resultant_peak_bw": round(float(max(np.max(val_r), np.max(val_l)) / bw), 2),
        "right_mean_n": round(float(np.mean(val_r)), 2),
        "right_peak_n": round(float(np.max(val_r)), 2),
        "left_mean_n": round(float(np.mean(val_l)), 2),
        "left_peak_n": round(float(np.max(val_l)), 2),
    }


def _format_knee_sections(
    reactions: JointReactionSeries, bw: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    tf: dict[str, Any] = {
        "joint_r": "walker_knee_r",
        "joint_l": "walker_knee_l",
        "frame": "resultant 3D spatial force norm",
    }
    tf.update(_bilateral_metrics(reactions.knee_tf_r_n, reactions.knee_tf_l_n, bw))
    pf: dict[str, Any] = {
        "model_has_patella_joint": True,
        "joint_r": "patellofemoral_r",
        "joint_l": "patellofemoral_l",
        "frame": "resultant 3D spatial force norm",
    }
    pf.update(_bilateral_metrics(reactions.knee_pf_r_n, reactions.knee_pf_l_n, bw))
    return tf, pf


def _format_shoulder_section(
    reactions: JointReactionSeries, bw: float
) -> dict[str, Any]:
    gh: dict[str, Any] = {
        "status": "complete",
        "is_articulated": True,
        "joint_r": "acromial_r",
        "joint_l": "acromial_l",
        "frame": "resultant 3D spatial force norm",
        "note": "acromial_r/l is articulated CustomJoint (3 rotational DOFs, not WeldJoint)",
    }
    gh.update(
        _bilateral_metrics(reactions.shoulder_gh_r_n, reactions.shoulder_gh_l_n, bw)
    )
    return gh


def format_joint_reaction_solution(
    lift: str,
    times: np.ndarray,
    body_weight_n: float,
    reactions: JointReactionSeries,
    phases: dict[str, dict[str, float]],
    cross_check: dict[str, Any] | None,
    static_bound: dict[str, Any],
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Format full JSON-serializable joint reaction receipt."""
    ctx = extra_context or {}
    bw = body_weight_n
    t0, t1 = float(times[0]), float(times[-1])
    bar_mass = ctx.get("bar_mass_kg")
    peak_lumbar = ctx.get("peak_lumbar_moment_nm")
    qual = ctx.get("qualification_info") or {}

    tf, pf = _format_knee_sections(reactions, bw)
    l5s1 = _format_l5s1_section(reactions, bw, peak_lumbar)
    shoulder = _format_shoulder_section(reactions, bw)

    return {
        "lift": lift,
        "status": "complete",
        "classification": qual.get("classification"),
        "qualification": (
            f"{qual.get('classification')}: inherited from the OSL-4 muscle solve "
            f"(reserve share threshold {qual.get('reserve_share_threshold')})"
        ),
        "reserve_share": qual.get("reserve_share"),
        "mean_reserve_share": qual.get("mean_reserve_share"),
        "peak_reserve_share": qual.get("peak_reserve_share"),
        "reserve_share_threshold": qual.get("reserve_share_threshold"),
        "window_start_s": round(t0, 4),
        "window_end_s": round(t1, 4),
        "duration_s": round(t1 - t0, 4),
        "body_weight_n": round(bw, 2),
        "lifter_mass_kg": round(bw / 9.80665, 2),
        "bar_mass_kg": None if bar_mass is None else round(float(bar_mass), 2),
        "acceleration_error_max": float(reactions.max_acc_error),
        "acceleration_realization": ctx.get("acceleration_realization"),
        "l5_s1": l5s1,
        "knee_tibiofemoral": tf,
        "knee_patellofemoral": pf,
        "shoulder_glenohumeral": shoulder,
        "cross_check_movement_optimizer": cross_check,
        "static_hold_analytic_bound": static_bound,
        "phases": phases,
        "timeseries": {
            "time_s": [round(float(v), 4) for v in times],
            "l5s1_comp_n": [round(float(v), 2) for v in reactions.l5s1_comp_n],
            "l5s1_ap_shear_n": [round(float(v), 2) for v in reactions.l5s1_ap_shear_n],
            "l5s1_torso_comp_n": [
                round(float(v), 2) for v in reactions.l5s1_torso_comp_n
            ],
            "l5s1_torso_ap_shear_n": [
                round(float(v), 2) for v in reactions.l5s1_torso_ap_shear_n
            ],
            "knee_tf_r_n": [round(float(v), 2) for v in reactions.knee_tf_r_n],
            "knee_tf_l_n": [round(float(v), 2) for v in reactions.knee_tf_l_n],
            "knee_pf_r_n": [round(float(v), 2) for v in reactions.knee_pf_r_n],
            "knee_pf_l_n": [round(float(v), 2) for v in reactions.knee_pf_l_n],
            "shoulder_gh_r_n": [round(float(v), 2) for v in reactions.shoulder_gh_r_n],
            "shoulder_gh_l_n": [round(float(v), 2) for v in reactions.shoulder_gh_l_n],
        },
    }


def _eval_mo_planar_curves(
    targets: Any,
    torso_angles: np.ndarray,
    opensim_torso_arms_kg: float,
    lift: str = "squat",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    tools_src = Path(__file__).resolve().parents[5] / "vendor" / "ud-tools" / "src"
    if str(tools_src) not in sys.path:
        sys.path.insert(0, str(tools_src))

    from movement_optimizer.models import BodyModel
    from movement_optimizer.spine_loads import (
        spinal_compression,
        spinal_shear,
        to_adr0055_ap_shear,
    )

    times = targets.time_s
    q_opt = np.zeros((len(times), 3))
    q_opt[:, 2] = torso_angles
    qd_opt = np.gradient(q_opt, times, axis=0)
    qdd_opt = np.gradient(qd_opt, times, axis=0)

    exercise_type = "deadlift" if lift == "deadlift" else "squat"
    body_default = BodyModel(body_mass=targets.body_mass_kg, height=targets.height_m)
    bar_mass = targets.bar_mass_kg
    comp_def = spinal_compression(
        q_opt, qd_opt, qdd_opt, body_default, bar_mass, exercise_type
    )
    shear_def = spinal_shear(
        q_opt, qd_opt, qdd_opt, body_default, bar_mass, exercise_type
    )
    adr_def = to_adr0055_ap_shear(shear_def)

    body_aligned = BodyModel(body_mass=targets.body_mass_kg, height=targets.height_m)
    if exercise_type == "deadlift":
        body_aligned.m_deadlift[2] = opensim_torso_arms_kg
        m_def = float(body_default.m_deadlift[2]) + bar_mass
        m_align = float(body_aligned.m_deadlift[2]) + bar_mass
    else:
        body_aligned.m_squat[2] = opensim_torso_arms_kg
        m_def = float(body_default.m_squat[2]) + bar_mass
        m_align = float(body_aligned.m_squat[2]) + bar_mass

    comp_align = spinal_compression(
        q_opt, qd_opt, qdd_opt, body_aligned, bar_mass, exercise_type
    )
    shear_align = spinal_shear(
        q_opt, qd_opt, qdd_opt, body_aligned, bar_mass, exercise_type
    )
    adr_align = to_adr0055_ap_shear(shear_align)

    return (
        comp_def,
        shear_def,
        adr_def,
        comp_align,
        shear_align,
        adr_align,
        m_def,
        m_align,
    )


def _difference_explanation(
    m_above: float,
    tol_n: float,
    mean_dc: float | None,
    max_dc: float | None,
    mean_ds: float | None,
    max_ds: float | None,
) -> str:
    """Describe the aligned residual from the computed values only."""
    base = (
        "Movement Optimizer is a planar rigid-body estimate (gravity and torso "
        "inertia, no muscles); OpenSim is a 3D JointReaction with no trunk muscles."
    )
    if mean_dc is None or mean_ds is None:
        return f"{base} No aligned OpenSim torso-frame reaction was supplied."
    return (
        f"{base} Rotated into the torso frame with matched m_above "
        f"({m_above:.2f} kg), the residual is {mean_dc:.2f} N compression "
        f"(max |{max_dc:.2f}| N) and {mean_ds:.2f} N AP shear (max |{max_ds:.2f}| N) "
        f"against the {tol_n:.1f} N tolerance. The residual is not yet decomposed "
        "into its sources (pelvis acceleration, 3D coupling, contact)."
    )


def crosscheck_movement_optimizer_l5s1(
    targets: Any,
    torso_angles: np.ndarray,
    torso_reactions: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    opensim_m_above_kg: float,
    lift: str = "squat",
) -> dict[str, Any]:
    """Compute Movement Optimizer spine_loads.py L5/S1 on lift kinematics.

    ``opensim_m_above_kg`` is the model mass above L5/S1 including the bar,
    summed from the OpenSim body set by the caller.
    """
    require(
        opensim_m_above_kg > targets.bar_mass_kg,
        "opensim_m_above_kg must exceed the bar mass",
    )
    (
        comp_def,
        shear_def,
        adr_def,
        comp_align,
        shear_align,
        adr_align,
        m_def,
        m_align,
    ) = _eval_mo_planar_curves(
        targets,
        torso_angles,
        opensim_m_above_kg - targets.bar_mass_kg,
        lift=lift,
    )

    tol_n = 50.0
    diff_comp_n = diff_shear_n = None
    mean_dc = max_dc = mean_ds = max_ds = None
    tol_passed: bool | None = None

    if torso_reactions is not None:
        osim_c, osim_s = torso_reactions
        dc = osim_c - comp_align
        ds = osim_s - adr_align
        diff_comp_n = [round(float(v), 2) for v in dc]
        diff_shear_n = [round(float(v), 2) for v in ds]
        mean_dc = round(float(np.mean(dc)), 2)
        max_dc = round(float(np.max(np.abs(dc))), 2)
        mean_ds = round(float(np.mean(ds)), 2)
        max_ds = round(float(np.max(np.abs(ds))), 2)
        tol_passed = bool(max_dc <= tol_n and max_ds <= tol_n)

    return {
        "model": "planar_ergonomic_l5s1_estimate",
        "is_joint_reaction": False,
        "comparison_frame": "torso frame (longitudinal spine axis, aligned like-for-like)",
        "pre_evaluation_tolerance_n": tol_n,
        "tolerance_satisfied": tol_passed,
        "opensim_m_above_kg": round(opensim_m_above_kg, 4),
        "movement_optimizer_m_above_default_kg": round(m_def, 4),
        "movement_optimizer_m_above_aligned_kg": round(m_align, 4),
        "opensim_angle_definition": "Sagittal tilt of torso frame relative to vertical in ground: atan2(R_torso[0, 1], R_torso[1, 1])",
        "movement_optimizer_angle_definition": "Torso inclination angle theta from vertical in 2D sagittal plane",
        "torso_angle_mean_deg": round(float(np.degrees(np.mean(torso_angles))), 2),
        "torso_angle_peak_deg": round(float(np.degrees(np.max(torso_angles))), 2),
        "compression_mean_n": round(float(np.mean(comp_def)), 2),
        "compression_peak_n": round(float(np.max(comp_def)), 2),
        "shear_mean_n": round(float(np.mean(shear_def)), 2),
        "shear_peak_n": round(float(np.max(shear_def)), 2),
        "adr0055_ap_shear_mean_n": round(float(np.mean(adr_def)), 2),
        "adr0055_ap_shear_peak_n": round(float(np.min(adr_def)), 2),
        "aligned_compression_mean_n": round(float(np.mean(comp_align)), 2),
        "aligned_compression_peak_n": round(float(np.max(comp_align)), 2),
        "aligned_adr0055_ap_shear_mean_n": round(float(np.mean(adr_align)), 2),
        "aligned_adr0055_ap_shear_peak_n": round(float(np.min(adr_align)), 2),
        "aligned_residual_compression_mean_n": mean_dc,
        "aligned_residual_compression_max_n": max_dc,
        "aligned_residual_ap_shear_mean_n": mean_ds,
        "aligned_residual_ap_shear_max_n": max_ds,
        "compression_curve_n": [round(float(v), 2) for v in comp_def],
        "adr0055_ap_shear_curve_n": [round(float(v), 2) for v in adr_def],
        "aligned_compression_curve_n": [round(float(v), 2) for v in comp_align],
        "aligned_adr0055_ap_shear_curve_n": [round(float(v), 2) for v in adr_align],
        "aligned_difference_per_frame": {
            "compression_diff_n": diff_comp_n,
            "shear_diff_n": diff_shear_n,
        },
        "model_difference_explanation": _difference_explanation(
            opensim_m_above_kg, tol_n, mean_dc, max_dc, mean_ds, max_ds
        ),
    }
