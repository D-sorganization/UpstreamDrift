"""Measure OpenSim joint reaction loads for the barbell lifts (OSL-5, #13043).

Usage (headless, from the repository root, real OpenSim 4.6)::

    UPSTREAMDRIFT_MSK_BASE_MODEL=$HOME/Repositories/ud-13044/shared/models/opensim/opensim-models/Models/Rajagopal/RajagopalLaiUhlrich2023.osim \\
    PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \\
        python3 scripts/measure_opensim_lift_joint_reactions.py \\
        --out docs/development/lifting/evidence/opensim_joint_reactions/receipts.json

Evaluates 3D joint reaction loads on the squat OSL-4 static-optimization solution:
- L5/S1 (back joint in inferior pelvis ISB frame and torso frame)
- Knee: tibiofemoral resultant, plus patellofemoral load (patellofemoral_r/l)
- Shoulder: glenohumeral (acromial_r/l articulated CustomJoints)
- Cross-check against Movement Optimizer spine_loads.py L5/S1
- Static-hold analytic bound verification

The other four lifts are recorded unavailable with explicit reasons (never fake values).
Renders at most two PNG plots (<300 KB each) into the evidence directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_SRC = ROOT / "vendor" / "ud-tools" / "src"
if str(TOOLS_SRC) not in sys.path:
    sys.path.insert(0, str(TOOLS_SRC))

from src.engines.physics_engines.opensim.python.msk_lift_joint_reactions import (  # noqa: E402
    evaluate_lift_joint_reactions,
)
from src.shared.python.lifting.pack_audit.names import LIFTS  # noqa: E402

logger = logging.getLogger("measure_opensim_lift_joint_reactions")
DEFAULT_OUT_DIR = ROOT / "docs/development/lifting/evidence/opensim_joint_reactions"
DEFAULT_RECEIPT = DEFAULT_OUT_DIR / "receipts.json"


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip()


def _shade_phase(ax: plt.Axes, t_start: float, t_end: float, label: str) -> None:
    ax.axvspan(t_start, t_end, color="#FED7AA", alpha=0.35)
    y_top = ax.get_ylim()[1]
    ax.text(
        0.5 * (t_start + t_end),
        y_top * 0.93,
        label,
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="semibold",
        color="#9A3412",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "#FFEDD5",
            "edgecolor": "#FDBA74",
            "alpha": 0.9,
        },
    )


def _plot_l5s1_loads(solution: dict[str, Any], out_path: Path) -> Path:
    ts = solution["timeseries"]
    times = np.asarray(ts["time_s"])
    t_start, t_end = float(times[0]), float(times[-1])
    comp_osim = np.asarray(ts["l5s1_comp_n"])
    shear_osim = np.asarray(ts["l5s1_ap_shear_n"])

    cc = solution.get("cross_check_movement_optimizer") or {}
    comp_opt = np.asarray(cc.get("compression_curve_n", []))
    shear_opt = np.asarray(cc.get("adr0055_ap_shear_curve_n", []))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.0), dpi=130, sharex=True)

    ax1.plot(
        times,
        comp_osim,
        label="OpenSim 3D JointReaction (+F_y)",
        color="#1E293B",
        lw=2.2,
    )
    if len(comp_opt) == len(times):
        ax1.plot(
            times,
            comp_opt,
            label="Movement Optimizer (Planar Ergonomic)",
            color="#2563EB",
            lw=1.8,
            ls="--",
        )
    ax1.set_ylabel("Compression (N)", fontsize=10)
    ax1.set_title(
        "Back Squat: L5/S1 Joint Reaction vs Movement Optimizer",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)
    _shade_phase(ax1, t_start, t_end, "bottom (0.00-0.20 s)")

    ax2.plot(
        times,
        shear_osim,
        label="OpenSim 3D JointReaction (+F_x)",
        color="#DC2626",
        lw=2.2,
    )
    if len(shear_opt) == len(times):
        ax2.plot(
            times,
            shear_opt,
            label="Movement Optimizer (ADR-0055 AP Shear)",
            color="#059669",
            lw=1.8,
            ls="--",
        )
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("AP Shear Force (N)", fontsize=10)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)
    _shade_phase(ax2, t_start, t_end, "bottom (0.00-0.20 s)")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_knee_and_shoulder(solution: dict[str, Any], out_path: Path) -> Path:
    ts = solution["timeseries"]
    times = np.asarray(ts["time_s"])
    t_start, t_end = float(times[0]), float(times[-1])
    tf_r, tf_l = np.asarray(ts["knee_tf_r_n"]), np.asarray(ts["knee_tf_l_n"])
    pf_r, pf_l = np.asarray(ts["knee_pf_r_n"]), np.asarray(ts["knee_pf_l_n"])
    gh_r, gh_l = np.asarray(ts["shoulder_gh_r_n"]), np.asarray(ts["shoulder_gh_l_n"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.0), dpi=130, sharex=True)

    ax1.plot(
        times,
        tf_r,
        label="Tibiofemoral Right (walker_knee_r)",
        color="#1E40AF",
        lw=2.0,
    )
    ax1.plot(
        times,
        tf_l,
        label="Tibiofemoral Left (walker_knee_l)",
        color="#3B82F6",
        lw=1.8,
        ls="--",
    )
    ax1.plot(
        times,
        pf_r,
        label="Patellofemoral Right (patellofemoral_r)",
        color="#B45309",
        lw=2.0,
    )
    ax1.plot(
        times,
        pf_l,
        label="Patellofemoral Left (patellofemoral_l)",
        color="#F59E0B",
        lw=1.8,
        ls="--",
    )
    ax1.set_ylabel("Contact Resultant (N)", fontsize=10)
    ax1.set_title(
        "Back Squat: Knee and Shoulder Joint Reactions (3D Spatial Resultant)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    _shade_phase(ax1, t_start, t_end, "bottom (0.00-0.20 s)")

    ax2.plot(
        times,
        gh_r,
        label="Glenohumeral Right (acromial_r)",
        color="#9333EA",
        lw=2.0,
    )
    ax2.plot(
        times,
        gh_l,
        label="Glenohumeral Left (acromial_l)",
        color="#C084FC",
        lw=1.8,
        ls="--",
    )
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylabel("Shoulder Reaction (N)", fontsize=10)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    _shade_phase(ax2, t_start, t_end, "bottom (0.00-0.20 s)")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _fmt_val(val: Any, fmt: str = ".2f", suffix: str = "") -> str:
    if val is None or isinstance(val, bool):
        return "unavailable"
    try:
        fval = float(val) if fmt != "d" else int(val)
        return f"{fval:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return "unavailable"


def _fmt_status(val: Any) -> str:
    if val is True:
        return "PASSED"
    if val is False:
        return "FAILED"
    return "unavailable"


def _render_header_section(squat: dict[str, Any], commit: str) -> str:
    w_start = squat.get("window_start_s")
    w_end = squat.get("window_end_s")
    ts = squat.get("timeseries")
    n_frames = len(ts["time_s"]) if isinstance(ts, dict) and "time_s" in ts else None
    if w_start is not None and w_end is not None:
        f_str = f"{n_frames} frames" if n_frames is not None else "unavailable"
        window_str = f"{_fmt_val(w_start)} s to {_fmt_val(w_end)} s ({f_str}, bottom inflection phase)"
    else:
        window_str = "unavailable"

    bw_val = squat.get("body_weight_n")
    bw_str = f" (Body Weight = {_fmt_val(bw_val)} N)" if bw_val is not None else ""
    bar_str = (
        f"{_fmt_val(squat.get('bar_mass_kg'))} kg"
        if squat.get("bar_mass_kg") is not None
        else "unavailable"
    )
    lifter_str = (
        f"{_fmt_val(squat.get('lifter_mass_kg'))} kg{bw_str}"
        if squat.get("lifter_mass_kg") is not None
        else "unavailable"
    )
    qual_cls = squat.get("classification") or "unavailable"

    return f"""# OpenSim Barbell Lift Joint Reaction Load Receipts (OSL-5, #13043)

`receipts.json` and evidence plots are generated by `scripts/measure_opensim_lift_joint_reactions.py`
using real OpenSim 4.6 and the Rajagopal 2023 musculoskeletal model.

## Overview

OSL-5 computes 3D internal joint reaction loads across the barbell lift motion window:
1. **L5/S1 Lumbar Joint**: OpenSim `back` joint expressed in inferior pelvis ISB frame and torso frame.
   *Qualification*: lower bound: no trunk muscles; the lumbar moment is an ideal actuator.
2. **Knee Tibiofemoral Joint**: Resultant contact load across `walker_knee_r` and `walker_knee_l`.
3. **Knee Patellofemoral Joint**: Articulated patella joint contact load (`patellofemoral_r/l`).
4. **Shoulder Glenohumeral Joint**: Evaluated at articulated `acromial_r/l` CustomJoints (3 rotational DOFs).
5. **Movement Optimizer Cross-Check**: Compares L5/S1 loads against `movement_optimizer.spine_loads` in aligned torso frame.
6. **Static-Hold Analytic Bound**: Frame/sign self-consistency at default pose and evaluation on dynamic squat frame.

## Evidence Summary (Back Squat Slice)

- **Source Commit**: `{commit}`
- **Lifter Mass**: {lifter_str}
- **Barbell Mass**: {bar_str}
- **Window**: {window_str}
- **Inherited OSL-4 Qualification**: {qual_cls} (mean reserve share: {_fmt_val(squat.get("mean_reserve_share"), ".4f")}, peak reserve share: {_fmt_val(squat.get("peak_reserve_share"), ".4f")}, threshold: {_fmt_val(squat.get("reserve_share_threshold"), ".2f")})"""


def _render_loads_table(squat: dict[str, Any]) -> str:
    l5s1 = squat.get("l5_s1") or {}
    tf = squat.get("knee_tibiofemoral") or {}
    pf = squat.get("knee_patellofemoral") or {}
    gh = squat.get("shoulder_glenohumeral") or {}
    qual = l5s1.get("qualification") or "unavailable"
    inh_qual = squat.get("classification") or "unavailable"

    lines = [
        "### Measured Reaction Loads\n",
        "| Joint | Metric | Qualification | Mean (N) | Peak (N) | Mean (BW) | Peak (BW) |",
        "|---|---|---|---|---|---|---|",
        f"| **L5/S1 Axial Compression** | +F_y (pelvis frame) | {qual} | {_fmt_val(l5s1.get('compression_mean_n'))} | {_fmt_val(l5s1.get('compression_peak_n'))} | {_fmt_val(l5s1.get('compression_mean_bw'))} | {_fmt_val(l5s1.get('compression_peak_bw'))} |",
        f"| **L5/S1 AP Shear** | +F_x (pelvis frame) | {qual} | {_fmt_val(l5s1.get('ap_shear_mean_n'))} | {_fmt_val(l5s1.get('ap_shear_peak_n'))} | {_fmt_val(l5s1.get('ap_shear_mean_bw'))} | {_fmt_val(l5s1.get('ap_shear_peak_bw'))} |",
        f"| **L5/S1 Torso Compression** | +F_y (torso frame) | {qual} | {_fmt_val(l5s1.get('compression_torso_frame_mean_n'))} | {_fmt_val(l5s1.get('compression_torso_frame_peak_n'))} | unavailable | unavailable |",
        f"| **L5/S1 Torso AP Shear** | +F_x (torso frame) | {qual} | {_fmt_val(l5s1.get('ap_shear_torso_frame_mean_n'))} | {_fmt_val(l5s1.get('ap_shear_torso_frame_peak_n'))} | unavailable | unavailable |",
        f"| **Knee Tibiofemoral** | Resultant 3D norm | {inh_qual} | {_fmt_val(tf.get('resultant_mean_n'))} | {_fmt_val(tf.get('resultant_peak_n'))} | {_fmt_val(tf.get('resultant_mean_bw'))} | {_fmt_val(tf.get('resultant_peak_bw'))} |",
        f"| **Knee Patellofemoral** | Resultant 3D norm | {inh_qual} | {_fmt_val(pf.get('resultant_mean_n'))} | {_fmt_val(pf.get('resultant_peak_n'))} | {_fmt_val(pf.get('resultant_mean_bw'))} | {_fmt_val(pf.get('resultant_peak_bw'))} |",
        f"| **Shoulder Glenohumeral** | Resultant 3D norm | {inh_qual} | {_fmt_val(gh.get('resultant_mean_n'))} | {_fmt_val(gh.get('resultant_peak_n'))} | {_fmt_val(gh.get('resultant_mean_bw'))} | {_fmt_val(gh.get('resultant_peak_bw'))} |",
        "",
        f"- **Peak Lumbar Actuator Moment**: {_fmt_val(l5s1.get('peak_lumbar_actuator_moment_nm'))} N·m (ideal coordinate actuator; no active erector spinae in Rajagopal)",
        f"- **Segmental Spine Reference**: {l5s1.get('segmental_spine_reference') or 'unavailable'}",
    ]
    return "\n".join(lines)


def _render_analytic_bound_section(squat: dict[str, Any]) -> str:
    bound = squat.get("static_hold_analytic_bound") or {}
    def_pose = bound.get("frame_sign_self_consistency_at_default_pose") or {}
    min_acc = bound.get("squat_min_acceleration_frame") or {}

    diff_str = (
        _fmt_val(def_pose.get("absolute_difference_n"), ".4e")
        if def_pose.get("absolute_difference_n") is not None
        else "unavailable"
    )

    return f"""### Static-Hold Analytic Bound & Dynamic Squat Frame Check

#### Default Pose Self-Consistency (Frame and Sign)
- **Upper Mass Above L5/S1**: {_fmt_val(def_pose.get("m_above_kg"), ".4f")} kg (Torso, Arms, Barbell)
- **Gravitational Weight Above ($W_{{\\text{{above}}}}$)**: {_fmt_val(def_pose.get("w_above_n"))} N
- **Measured L5/S1 Compression**: {_fmt_val(def_pose.get("l5_s1_compression_n"))} N
- **Absolute Difference**: {diff_str} N
- **Muscles Crossing L5/S1**: {_fmt_val(def_pose.get("muscles_crossing_l5s1_count"), "d")} (sum crossing forces = {_fmt_val(def_pose.get("sum_muscle_forces_crossing_l5s1_n"), ".4f")} N)
- **Verification Status**: **{_fmt_status(def_pose.get("passed"))}** ({def_pose.get("note") or "unavailable"})

#### Dynamic Squat Minimal-Acceleration Frame Evaluation
- **Frame Index**: {_fmt_val(min_acc.get("frame_index"), "d")} (time = {_fmt_val(min_acc.get("time_s"), ".4f")} s)
- **Generalized Acceleration Norm ($\\|\\ddot{{q}}\\|$)**: {_fmt_val(min_acc.get("qdd_norm_rad_s2"), ".4f")} rad/s²
- **Upper Weight Above ($W_{{\\text{{above}}}}$)**: {_fmt_val(min_acc.get("w_above_n"))} N
- **Measured Pelvis Compression**: {_fmt_val(min_acc.get("l5_s1_compression_pelvis_n"))} N
- **Measured Torso Compression**: {_fmt_val(min_acc.get("l5_s1_compression_torso_n"))} N
- **Lower Bound ($W_{{\\text{{above}}}} \\le \\text{{Compression}}$)**: {_fmt_status(min_acc.get("lower_bound_satisfied"))}
- **Upper Bound ($\\text{{Compression}} \\le W_{{\\text{{above}}}} + \\sum F_m$)**: {_fmt_status(min_acc.get("upper_bound_satisfied"))}
- **Evaluation Status**: **{_fmt_status(min_acc.get("passed"))}** ({min_acc.get("note") or "unavailable"})"""


def _render_mo_section(squat: dict[str, Any]) -> str:
    cc = squat.get("cross_check_movement_optimizer") or {}
    return f"""### Movement Optimizer L5/S1 Cross-Check

- **Model**: {cc.get("model") or "unavailable"}
- **Upper Body Mass ($m_{{\\text{{above}}}}$)**: OpenSim = {_fmt_val(cc.get("opensim_m_above_kg"), ".4f")} kg, Movement Optimizer default = {_fmt_val(cc.get("movement_optimizer_m_above_default_kg"), ".4f")} kg, Movement Optimizer aligned = {_fmt_val(cc.get("movement_optimizer_m_above_aligned_kg"), ".4f")} kg
- **Angle Definitions**:
  - OpenSim: {cc.get("opensim_angle_definition") or "unavailable"}
  - Movement Optimizer: {cc.get("movement_optimizer_angle_definition") or "unavailable"}
- **Default Movement Optimizer**: Compression Mean = {_fmt_val(cc.get("compression_mean_n"))} N, Peak = {_fmt_val(cc.get("compression_peak_n"))} N; ADR-0055 AP Shear Mean = {_fmt_val(cc.get("adr0055_ap_shear_mean_n"))} N, Peak = {_fmt_val(cc.get("adr0055_ap_shear_peak_n"))} N
- **Aligned Movement Optimizer (Torso Frame)**: Compression Mean = {_fmt_val(cc.get("aligned_compression_mean_n"))} N, Peak = {_fmt_val(cc.get("aligned_compression_peak_n"))} N; ADR-0055 AP Shear Mean = {_fmt_val(cc.get("aligned_adr0055_ap_shear_mean_n"))} N, Peak = {_fmt_val(cc.get("aligned_adr0055_ap_shear_peak_n"))} N
- **Pre-Evaluation Tolerance**: {_fmt_val(cc.get("pre_evaluation_tolerance_n"))} N
- **Residual Differences (OpenSim Torso vs Aligned Movement Optimizer)**:
  - Compression Mean Residual: {_fmt_val(cc.get("aligned_residual_compression_mean_n"))} N (Peak: {_fmt_val(cc.get("aligned_residual_compression_max_n"))} N)
  - AP Shear Mean Residual: {_fmt_val(cc.get("aligned_residual_ap_shear_mean_n"))} N (Peak: {_fmt_val(cc.get("aligned_residual_ap_shear_max_n"))} N)
- **Model Difference Analysis**: {cc.get("model_difference_explanation") or "unavailable"}"""


def _render_other_lifts_section(receipts: dict[str, Any]) -> str:
    other_reasons = set()
    for lk in ("deadlift", "bench_press", "snatch", "clean_and_jerk"):
        rec = receipts.get(lk)
        if isinstance(rec, dict) and rec.get("reason"):
            other_reasons.add(rec["reason"])
    other_reason_str = (
        "; ".join(sorted(other_reasons)) if other_reasons else "unavailable"
    )

    return f"""### Status of Other Four Lifts

The other four lifts (`deadlift`, `bench_press`, `snatch`, `clean_and_jerk`) are reported as `unavailable`
with reason: `{other_reason_str}`. Values are explicitly `null` (never synthetic zeros)."""


def _render_footer_section() -> str:
    return """## Evidence Figures

### L5/S1 Joint Reaction Loads vs Movement Optimizer
![L5/S1 Joint Reaction Loads](squat_l5s1_joint_reaction_loads.png)

### Knee and Shoulder Joint Reactions
![Knee and Shoulder Joint Reactions](squat_knee_and_shoulder_joint_reactions.png)

## Reproduction Commands

```bash
UPSTREAMDRIFT_MSK_BASE_MODEL=$HOME/Repositories/ud-13044/shared/models/opensim/opensim-models/Models/Rajagopal/RajagopalLaiUhlrich2023.osim \\
PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \\
    ~/Repositories/UpstreamDrift/.venv/bin/python3 scripts/measure_opensim_lift_joint_reactions.py
```"""


def render_evidence_readme(receipts: dict[str, Any], commit: str) -> str:
    """Render comprehensive Markdown evidence README from measured receipts."""
    squat = receipts.get("squat") or {}
    sections = [
        _render_header_section(squat, commit),
        _render_loads_table(squat),
        _render_analytic_bound_section(squat),
        _render_mo_section(squat),
        _render_other_lifts_section(receipts),
        _render_footer_section(),
    ]
    return "\n\n".join(sections) + "\n"


def _write_evidence_readme(
    out_dir: Path,
    receipts: dict[str, Any],
    commit: str,
) -> None:
    content = render_evidence_readme(receipts, commit)
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_RECEIPT, help="Output receipts.json path"
    )
    parser.add_argument(
        "--stride", type=int, default=5, help="Frame subsampling stride"
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for plots",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    commit = _git("rev-parse", "HEAD")
    logger.info("Computing lift joint reactions on commit %s", commit)

    receipts: dict[str, Any] = {}
    for lift in LIFTS:
        logger.info("Evaluating joint reactions for lift: %s", lift)
        sol = evaluate_lift_joint_reactions(lift, stride=args.stride)
        receipts[lift] = sol

    squat_sol = receipts["squat"]
    if squat_sol.get("status") == "complete" and "timeseries" in squat_sol:
        plot1 = args.plot_dir / "squat_l5s1_joint_reaction_loads.png"
        plot2 = args.plot_dir / "squat_knee_and_shoulder_joint_reactions.png"
        _plot_l5s1_loads(squat_sol, plot1)
        _plot_knee_and_shoulder(squat_sol, plot2)
        logger.info("Saved plot 1: %s (%d bytes)", plot1, plot1.stat().st_size)
        logger.info("Saved plot 2: %s (%d bytes)", plot2, plot2.stat().st_size)

    full_receipt = {
        "generator": "scripts/measure_opensim_lift_joint_reactions.py",
        "issue": "#13043 (OSL-5)",
        "source_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_source": "RajagopalLaiUhlrich2023",
        "l5_s1_joint": "back",
        "l5_s1_frame": "inferior vertebra ISB frame (pelvis, X anterior, Y cranial, Z right)",
        "knee_tibiofemoral_joints": ["walker_knee_r", "walker_knee_l"],
        "knee_patellofemoral_joints": ["patellofemoral_r", "patellofemoral_l"],
        "shoulder_glenohumeral_joints": ["acromial_r", "acromial_l"],
        "lifts": receipts,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(full_receipt, f, indent=2)
    logger.info("Wrote receipt: %s", args.out)

    _write_evidence_readme(args.plot_dir, receipts, commit)
    logger.info("Wrote evidence README: %s", args.plot_dir / "README.md")


if __name__ == "__main__":
    main()
