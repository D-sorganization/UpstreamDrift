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
from collections.abc import Sequence
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


# Color palette for phases
PHASE_COLORS: dict[str, str] = {
    "standing": "#E2E8F0",
    "descent": "#CBD5E1",
    "bottom": "#FED7AA",
    "ascent": "#BAE6FD",
    "lockout": "#E2E8F0",
}


def _shade_phases(
    ax: plt.Axes,
    solution: dict[str, Any],
    t_start: float,
    t_end: float,
    lift: str = "squat",
) -> None:
    phase_dict = solution.get("phases") or {}
    first_key = list(phase_dict.keys())[0] if phase_dict else ""
    if len(phase_dict) == 1 and (
        "bottom" in first_key or "pull" in first_key or "engine window" in first_key
    ):
        seg_label = first_key
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
        phases = lift_phases(lift)
        for p_idx, p in enumerate(phases):
            t_p0 = t_start + p.t_frac * (t_end - t_start)
            t_p1 = (
                t_start + phases[p_idx + 1].t_frac * (t_end - t_start)
                if p_idx + 1 < len(phases)
                else t_end
            )
            col = PHASE_COLORS.get(p.name, "#F1F5F9")
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


def _plot_activations(solution: dict[str, Any], out_path: Path) -> Path:
    times = np.asarray(solution["times"])
    t_start, t_end = float(times[0]), float(times[-1])
    activations = np.asarray(solution["activations"])
    muscle_names = solution["muscle_names"]
    name_to_idx = {name: i for i, name in enumerate(muscle_names)}
    lift = solution.get("lift", "squat")
    title_lift = "Back Squat" if lift == "squat" else lift.replace("_", " ").title()

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
        f"{title_lift}: Muscle Group Activations (Engine Window {t_start:.1f}-{t_end:.1f} s)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(-0.02, 1.05)
    _shade_phases(ax, solution, t_start, t_end, lift=lift)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_forces(solution: dict[str, Any], out_path: Path) -> Path:
    times = np.asarray(solution["times"])
    t_start, t_end = float(times[0]), float(times[-1])
    forces = np.asarray(solution["forces"])
    muscle_names = solution["muscle_names"]
    name_to_idx = {name: i for i, name in enumerate(muscle_names)}
    lift = solution.get("lift", "squat")
    title_lift = "Back Squat" if lift == "squat" else lift.replace("_", " ").title()

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
        f"{title_lift}: Total Muscle Force per Group (Engine Window {t_start:.1f}-{t_end:.1f} s)",
        fontsize=11,
        fontweight="bold",
    )
    _shade_phases(ax, solution, t_start, t_end, lift=lift)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_moments_and_reserves(solution: dict[str, Any], out_path: Path) -> Path:
    times = np.asarray(solution["times"])
    t_start, t_end = float(times[0]), float(times[-1])
    tau_targets = np.asarray(solution["tau_targets"])
    tau_muscles = np.asarray(solution["tau_muscles"])
    reserves = np.asarray(solution["reserves"])
    lift = solution.get("lift", "squat")
    title_lift = "Back Squat" if lift == "squat" else lift.replace("_", " ").title()

    total_demand = np.sum(np.abs(tau_targets), axis=1)
    total_muscle = np.sum(np.abs(tau_muscles), axis=1)
    total_reserve = np.sum(np.abs(reserves), axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
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
        f"{title_lift}: Leg Moment Balance and Reserve Actuator Load (Share: {solution['reserve_share']:.1%})",
        fontsize=11,
        fontweight="bold",
    )
    _shade_phases(ax, solution, t_start, t_end, lift=lift)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close(fig)
    return out_path


def _render_lift_plots(solution: dict[str, Any], lift: str, out_dir: Path) -> list[str]:
    """Generate phase-resolved PNG plots under 300 KB each for *lift*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        _plot_activations(
            solution, out_dir / f"{lift}_muscle_activations_by_phase.png"
        ).name,
        _plot_forces(solution, out_dir / f"{lift}_muscle_forces_by_phase.png").name,
        _plot_moments_and_reserves(
            solution, out_dir / f"{lift}_joint_moments_and_reserves.png"
        ).name,
    ]

    # Verify size constraint (< 300 KB)
    for fname in generated:
        fpath = out_dir / fname
        size_kb = fpath.stat().st_size / 1024
        logger.info("Generated %s (%.1f KB)", fname, size_kb)
        if size_kb >= 300.0:
            raise RuntimeError(f"Plot {fname} exceeded 300 KB budget: {size_kb:.1f} KB")

    return generated


def _render_squat_plots(solution: dict[str, Any], out_dir: Path) -> list[str]:
    """Backwards-compatible alias for rendering squat plots."""
    return _render_lift_plots(solution, "squat", out_dir)


def measure(
    stride: int = 5,
    out_dir: Path = DEFAULT_OUT_DIR,
    target_lifts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate muscle activations and forces for barbell lifts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts_file = out_dir / "receipts.json"
    existing_lifts: dict[str, Any] = {}
    if receipts_file.is_file():
        try:
            existing_data = json.loads(receipts_file.read_text(encoding="utf-8"))
            existing_lifts = existing_data.get("lifts", {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read existing receipts: %s", exc)

    eval_lifts = target_lifts if target_lifts is not None else LIFTS
    receipts: dict[str, Any] = dict(existing_lifts)

    for lift in eval_lifts:
        logger.info("Evaluating muscles for %s...", lift)
        res = evaluate_lift_muscles(lift, stride=stride, run_moco=True)
        if res["status"] == "complete":
            _render_lift_plots(res, lift, out_dir)
            entry: dict[str, Any] = {
                "lift": lift,
                "status": "complete",
                "classification": res["classification"],
                "reserve_share": res["reserve_share"],
                "mean_reserve_share": res["mean_reserve_share"],
                "peak_reserve_share": res["peak_reserve_share"],
                "residual_status": res["residual_status"],
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
            if res.get("external_loads"):
                entry["external_loads"] = res["external_loads"]
            if res.get("limitations"):
                entry["limitations"] = res["limitations"]
            receipts[lift] = entry
        else:
            receipts[lift] = {
                "lift": lift,
                "status": "unavailable",
                "classification": "unqualified",
                "reason": res.get("reason", "unknown reason"),
                "reserve_share": None,
                "mean_reserve_share": None,
                "peak_reserve_share": None,
                "residual_status": res.get("residual_status"),
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

    ordered_receipts = {k: receipts[k] for k in LIFTS if k in receipts}
    for k in receipts:
        if k not in ordered_receipts:
            ordered_receipts[k] = receipts[k]

    return {
        "generator": "scripts/measure_opensim_lift_muscles.py",
        "issue": "#13042 (OSL-4)",
        "source_commit": _git("rev-parse", "HEAD"),
        "reserve_share_threshold": RESERVE_SHARE_THRESHOLD,
        "citation": RAJAGOPAL_MODEL_CITATION,
        "lifts": ordered_receipts,
    }


LIFT_DISPLAY_NAMES: dict[str, str] = {
    "squat": "Squat",
    "deadlift": "Deadlift",
    "bench_press": "Bench Press",
    "snatch": "Snatch",
    "clean_and_jerk": "Clean & Jerk",
}


def render_per_lift_table(receipt: dict[str, Any]) -> str:
    """Render a markdown summary table for all barbell lifts from receipts."""
    lifts_data: dict[str, Any] = receipt.get("lifts", receipt)
    ordered_keys = [lift for lift in LIFTS if lift in lifts_data]
    extra_keys = [lift for lift in lifts_data if lift not in ordered_keys]
    target_lifts = ordered_keys + extra_keys
    if not target_lifts:
        target_lifts = list(lifts_data.keys())

    header = (
        "| Lift | Status | Classification | Window (s) | Mean Reserve Share | "
        "Peak Reserve Share | Pelvis Residual Force (N, vs Hicks) | "
        "Pelvis Residual Moment (N·m, vs Hicks) | Within / Outside |"
    )
    separator = "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    rows: list[str] = [header, separator]

    for lift in target_lifts:
        rec = lifts_data.get(lift)
        if not isinstance(rec, dict):
            rec = {}
        name = LIFT_DISPLAY_NAMES.get(lift, lift.replace("_", " ").title())
        status = rec.get("status")
        status_str = f"`{status}`" if status else "unavailable"
        classification = rec.get("classification")
        class_str = f"`{classification}`" if classification else "unavailable"

        w_start = rec.get("window_start_s")
        w_end = rec.get("window_end_s")
        if w_start is not None and w_end is not None:
            window_str = f"{w_start:.1f}–{w_end:.1f}"
        else:
            window_str = "unavailable"

        mean_share = rec.get("mean_reserve_share")
        if mean_share is None and isinstance(rec.get("reserve_share"), (int, float)):
            mean_share = rec.get("reserve_share")
        mean_str = (
            f"{mean_share * 100:.1f}%" if mean_share is not None else "unavailable"
        )

        peak_share = rec.get("peak_reserve_share")
        peak_str = (
            f"{peak_share * 100:.1f}%" if peak_share is not None else "unavailable"
        )

        res = rec.get("residual_status")
        if (
            isinstance(res, dict)
            and res.get("measured_pelvis_residual_force_n") is not None
            and res.get("hicks_force_limit_n") is not None
        ):
            meas_f = float(res["measured_pelvis_residual_force_n"])
            lim_f = float(res["hicks_force_limit_n"])
            force_str = f"{meas_f:.2f} (vs {lim_f:.2f})"
        else:
            force_str = "unavailable"

        if (
            isinstance(res, dict)
            and res.get("measured_pelvis_residual_moment_nm") is not None
            and res.get("hicks_moment_limit_nm") is not None
        ):
            meas_m = float(res["measured_pelvis_residual_moment_nm"])
            lim_m = float(res["hicks_moment_limit_nm"])
            moment_str = f"{meas_m:.2f} (vs {lim_m:.2f})"
        else:
            moment_str = "unavailable"

        if (
            isinstance(res, dict)
            and res.get("forces_within") is not None
            and res.get("moments_within") is not None
        ):
            within_str = (
                "Within"
                if (res["forces_within"] and res["moments_within"])
                else "Outside"
            )
        else:
            within_str = "unavailable"

        rows.append(
            f"| {name} | {status_str} | {class_str} | {window_str} | {mean_str} | "
            f"{peak_str} | {force_str} | {moment_str} | {within_str} |"
        )

    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--table", action="store_true", help="Print Markdown summary table"
    )
    parser.add_argument(
        "--lifts",
        nargs="+",
        choices=LIFTS,
        default=None,
        help="Specific lifts to evaluate (defaults to all)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_file = args.out
    out_dir = out_file.parent

    if args.table:
        if not out_file.is_file():
            logger.error("Receipts file not found: %s", out_file)
            return 1
        data = json.loads(out_file.read_text(encoding="utf-8"))
        print(render_per_lift_table(data))
        return 0

    record = measure(stride=args.stride, out_dir=out_dir, target_lifts=args.lifts)

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
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Could not run prettier: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
