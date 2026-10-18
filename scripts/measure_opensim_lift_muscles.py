"""Measure OpenSim muscle forces and activations for the barbell lifts (OSL-4, #13042).

Usage (headless, from the repository root, real OpenSim 4.6)::

    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
        python3 scripts/measure_opensim_lift_muscles.py \
        --out docs/development/lifting/evidence/opensim_muscles/receipts.json

For each of the five barbell lifts, the receipt records muscle activations, forces,
and reserve shares.  Squat is evaluated across all 80 lower-limb muscles, with
major group activations/forces summarized per lift phase and plots rendered into
docs/development/lifting/evidence/opensim_muscles/.
The other four lifts are recorded unavailable with reasons (never as zero).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure headless matplotlib
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.msk_lift_muscles import (  # noqa: E402
    GROUP_TO_RAJAGOPAL_MUSCLES,
    MAJOR_MUSCLE_GROUPS,
    RAJAGOPAL_MODEL_CITATION,
    RESERVE_SHARE_THRESHOLD,
    evaluate_lift_muscles,
)
from src.shared.python.lifting.lift_spec import lift_phases  # noqa: E402
from src.shared.python.lifting.pack_audit.names import LIFTS  # noqa: E402

logger = logging.getLogger("measure_opensim_lift_muscles")
DEFAULT_OUT_DIR = ROOT / "docs/development/lifting/evidence/opensim_muscles"
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


def _render_squat_plots(solution: dict[str, Any], out_dir: Path) -> list[str]:
    """Generate phase-resolved PNG plots under 300 KB each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    times = np.asarray(solution["times"])
    t_start, t_end = float(times[0]), float(times[-1])
    phases = lift_phases("squat")
    activations = np.asarray(solution["activations"])
    forces = np.asarray(solution["forces"])
    muscle_names = solution["muscle_names"]
    name_to_idx = {name: i for i, name in enumerate(muscle_names)}

    # Color palette for phases
    phase_colors = {
        "standing": "#E2E8F0",
        "descent": "#CBD5E1",
        "bottom": "#FED7AA",
        "ascent": "#BAE6FD",
        "lockout": "#E2E8F0",
    }

    def shade_phases(ax: plt.Axes) -> None:
        phase_dict = solution.get("phases") or {}
        if len(phase_dict) == 1 and "bottom" in list(phase_dict.keys())[0]:
            seg_label = list(phase_dict.keys())[0]
            ax.axvspan(t_start, t_end, color="#FED7AA", alpha=0.35)
            y_top = ax.get_ylim()[1]
            ax.text(
                0.5 * (t_start + t_end),
                y_top * 0.93,
                seg_label,
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
        else:
            for p_idx, p in enumerate(phases):
                t_p0 = t_start + p.t_frac * (t_end - t_start)
                t_p1 = (
                    t_start + phases[p_idx + 1].t_frac * (t_end - t_start)
                    if p_idx + 1 < len(phases)
                    else t_end
                )
                col = phase_colors.get(p.name, "#F1F5F9")
                ax.axvspan(
                    t_p0,
                    t_p1,
                    color=col,
                    alpha=0.5,
                    label=f"Phase: {p.name}" if p_idx == 0 else None,
                )
                ax.text(
                    0.5 * (t_p0 + t_p1),
                    ax.get_ylim()[1] * 0.92,
                    p.name,
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="semibold",
                    alpha=0.7,
                )

    # 1. Major Groups Activations by Phase
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    for group_name, col in (
        ("quadriceps", "#2563EB"),
        ("gluteals", "#DC2626"),
        ("hamstrings", "#059669"),
    ):
        muscles = GROUP_TO_RAJAGOPAL_MUSCLES[group_name]
        idxs = [name_to_idx[m] for m in muscles if m in name_to_idx]
        if idxs:
            mean_act = np.mean(activations[:, idxs], axis=1)
            ax.plot(
                times,
                mean_act,
                label=f"{group_name.capitalize()} (mean)",
                color=col,
                lw=2.0,
            )

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Activation [0, 1]", fontsize=10)
    ax.set_title(
        "Back Squat: Muscle Group Activations (Engine Window 0.0-0.2 s)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(-0.02, 1.05)
    shade_phases(ax)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()

    p1 = out_dir / "squat_muscle_activations_by_phase.png"
    plt.savefig(p1)
    plt.close(fig)
    generated.append(p1.name)

    # 2. Major Groups Forces by Phase
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    for group_name, col in (
        ("quadriceps", "#2563EB"),
        ("gluteals", "#DC2626"),
        ("hamstrings", "#059669"),
    ):
        muscles = GROUP_TO_RAJAGOPAL_MUSCLES[group_name]
        idxs = [name_to_idx[m] for m in muscles if m in name_to_idx]
        if idxs:
            sum_force = np.sum(forces[:, idxs], axis=1)
            ax.plot(
                times,
                sum_force,
                label=f"{group_name.capitalize()} (total)",
                color=col,
                lw=2.0,
            )

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Muscle Force (N)", fontsize=10)
    ax.set_title(
        "Back Squat: Total Muscle Force per Group (Engine Window 0.0-0.2 s)",
        fontsize=11,
        fontweight="bold",
    )
    shade_phases(ax)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()

    p2 = out_dir / "squat_muscle_forces_by_phase.png"
    plt.savefig(p2)
    plt.close(fig)
    generated.append(p2.name)

    # 3. Joint Moments vs Reserves
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    tau_targets = np.asarray(solution["tau_targets"])
    tau_muscles = np.asarray(solution["tau_muscles"])
    reserves = np.asarray(solution["reserves"])

    total_demand = np.sum(np.abs(tau_targets), axis=1)
    total_muscle = np.sum(np.abs(tau_muscles), axis=1)
    total_reserve = np.sum(np.abs(reserves), axis=1)

    ax.plot(
        times,
        total_demand,
        label="Total ID Joint Demand (sum |tau|)",
        color="#1E293B",
        lw=2.0,
    )
    ax.plot(
        times,
        total_muscle,
        label="Muscle-Generated Moment",
        color="#2563EB",
        lw=1.8,
        ls="--",
    )
    ax.plot(
        times,
        total_reserve,
        label="Reserve Actuator Moment",
        color="#DC2626",
        lw=1.8,
        ls="-.",
    )

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Joint Moment Magnitude (N m)", fontsize=10)
    ax.set_title(
        f"Back Squat: Leg Moment Balance and Reserve Actuator Load (Share: {solution['reserve_share']:.1%})",
        fontsize=11,
        fontweight="bold",
    )
    shade_phases(ax)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()

    p3 = out_dir / "squat_joint_moments_and_reserves.png"
    plt.savefig(p3)
    plt.close(fig)
    generated.append(p3.name)

    # Verify size constraint (< 300 KB)
    for fname in generated:
        fpath = out_dir / fname
        size_kb = fpath.stat().st_size / 1024
        logger.info("Generated %s (%.1f KB)", fname, size_kb)
        if size_kb >= 300.0:
            raise RuntimeError(f"Plot {fname} exceeded 300 KB budget: {size_kb:.1f} KB")

    return generated


def measure(stride: int = 5, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    """Evaluate muscle activations and forces for all five lifts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, Any] = {}

    for lift in LIFTS:
        logger.info("Evaluating muscles for %s...", lift)
        res = evaluate_lift_muscles(lift, stride=stride, run_moco=True)
        if lift == "squat" and res["status"] == "complete":
            _render_squat_plots(res, out_dir)
            receipts[lift] = {
                "lift": lift,
                "status": "complete",
                "classification": res["classification"],
                "reserve_share": res["reserve_share"],
                "mean_reserve_share": res["mean_reserve_share"],
                "peak_reserve_share": res["peak_reserve_share"],
                "reserve_share_threshold": RESERVE_SHARE_THRESHOLD,
                "window_start_s": res["window_start_s"],
                "window_end_s": res["window_end_s"],
                "duration_s": res["duration_s"],
                "start_pose": res["start_pose"],
                "coordinate_breakdown": res["coordinate_breakdown"],
                "primary_reserve_drivers": res["primary_reserve_drivers"],
                "bound_violations_or_pinned": res["bound_violations_or_pinned"],
                "hip_demand_asymmetry": res["hip_demand_asymmetry"],
                "moment_balance_max_error_nm": res["moment_balance_max_error_nm"],
                "frames": int(len(res["times"])),
                "stride": stride,
                "major_groups": res["major_groups"],
                "phases": res["phases"],
                "moco": res["moco"],
                "citation": res["citation"],
            }
        else:
            receipts[lift] = {
                "lift": lift,
                "status": "unavailable",
                "classification": "unqualified",
                "reason": res.get("reason", "unknown reason"),
                "reserve_share": None,
                "window_start_s": None,
                "window_end_s": None,
                "duration_s": None,
                "start_pose": None,
                "coordinate_breakdown": None,
                "primary_reserve_drivers": None,
                "bound_violations_or_pinned": None,
                "hip_demand_asymmetry": None,
                "major_groups": None,
                "phases": None,
                "moco": None,
            }

    return {
        "generator": "scripts/measure_opensim_lift_muscles.py",
        "issue": "#13042 (OSL-4)",
        "source_commit": _git("rev-parse", "HEAD"),
        "reserve_share_threshold": RESERVE_SHARE_THRESHOLD,
        "citation": RAJAGOPAL_MODEL_CITATION,
        "lifts": receipts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_file = args.out
    out_dir = out_file.parent

    record = measure(stride=args.stride, out_dir=out_dir)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Receipt written to %s", out_file)

    # Format JSON evidence with prettier
    try:
        subprocess.run(
            ["npx", "--yes", "prettier@3", "--write", str(out_file)],
            check=True,
            timeout=30,
        )
        logger.info("Formatted %s with prettier@3", out_file)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not run prettier: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
