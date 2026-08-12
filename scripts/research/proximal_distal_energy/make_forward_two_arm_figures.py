"""Render publication figures for the forward constrained two-hand study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

from scripts.research.proximal_distal_energy.two_arm_closed_loop import (  # noqa: E402
    TwoArmParams,
    kinematics,
)

FloatArray = npt.NDArray[np.float64]
OUTPUT_ROOT = Path("docs/research/proximal_distal_energy_transfer")
DATA_DIR = OUTPUT_ROOT / "data"
FIGURE_DIR = OUTPUT_ROOT / "figures"
FIGURE_STEMS = (
    "fig_forward_two_hand_couple_killswitch",
    "fig_forward_two_hand_force_geometry",
    "fig_forward_two_hand_numerical_audit",
)
COLORS = {
    "force": "#0F766E",
    "wrist": "#B45309",
    "branch": "#7C3AED",
    "right": "#0369A1",
    "left": "#BE123C",
    "energy": "#1D4ED8",
    "work": "#C2410C",
    "neutral": "#475569",
}


def _load_study(
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    json_path = data_dir / "forward_two_arm_study.json"
    npz_path = data_dir / "forward_two_arm_study.npz"
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            "forward two-hand JSON and NPZ evidence must both exist"
        )
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "forward-two-arm-evidence-v1":
        raise ValueError("unsupported forward two-hand evidence schema")
    with np.load(npz_path) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    required = {
        "baseline_time_s",
        "baseline_q",
        "baseline_contact_force_on_club_n",
        "baseline_force_generated_couple_nm",
        "baseline_direct_wrist_torque_nm",
        "baseline_control_power_w",
        "baseline_mechanical_energy_j",
        "baseline_position_constraint_norm_m",
        "branch_time_s",
        "branch_force_generated_couple_nm",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"evidence arrays are incomplete: {missing}")
    return record, arrays


def _params(record: dict[str, Any]) -> TwoArmParams:
    values = dict(record["parameters"])
    values["right_shoulder_m"] = tuple(values["right_shoulder_m"])
    values["left_shoulder_m"] = tuple(values["left_shoulder_m"])
    return TwoArmParams(**values)


def _save(figure: plt.Figure, figure_dir: Path, stem: str) -> tuple[Path, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / f"{stem}.pdf"
    svg_path = figure_dir / f"{stem}.svg"
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "UpstreamDrift Open Research", "CreationDate": None},
    )
    figure.savefig(svg_path, bbox_inches="tight", metadata={"Creator": "UpstreamDrift"})
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(figure)
    return pdf_path, svg_path


def _couple_killswitch_figure(
    record: dict[str, Any], arrays: dict[str, FloatArray]
) -> plt.Figure:
    time = arrays["baseline_time_s"]
    cut_index = int(record["representative_killswitch"]["cut_index"])
    branch_time = arrays["branch_time_s"]
    branch_relative_ms = 1_000.0 * (branch_time - branch_time[0])
    comparison_end = cut_index + branch_time.size

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(
        1_000.0 * time,
        arrays["baseline_force_generated_couple_nm"],
        color=COLORS["force"],
        linewidth=2.2,
        label="Force-Generated Couple",
    )
    axis.plot(
        1_000.0 * time,
        arrays["baseline_direct_wrist_torque_nm"],
        color=COLORS["wrist"],
        linewidth=1.7,
        label="Direct Wrist Torque",
    )
    onset = record["baseline"]["force_generated_couple"]["first_negative_time_s"]
    axis.axvline(1_000.0 * onset, color=COLORS["neutral"], linestyle="--")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Separated Club-Couple Contributions",
        xlabel="Time (ms)",
        ylabel="Moment (N m)",
    )
    axis.legend(frameon=False)

    axis = axes[0, 1]
    continued = arrays["baseline_force_generated_couple_nm"][cut_index:comparison_end]
    axis.plot(
        branch_relative_ms,
        continued,
        color=COLORS["force"],
        linewidth=2.0,
        label="Commanded Continuation",
    )
    axis.plot(
        branch_relative_ms,
        arrays["branch_force_generated_couple_nm"],
        color=COLORS["branch"],
        linewidth=2.2,
        label="Zero-Command Branch",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Matched-State Torque Killswitch",
        xlabel="Time After 200 ms Cut (ms)",
        ylabel="Force-Generated Couple (N m)",
    )
    axis.legend(frameon=False)

    axis = axes[1, 0]
    axis.plot(
        1_000.0 * time,
        arrays["baseline_control_power_w"],
        color=COLORS["work"],
        linewidth=1.8,
        label="Applied Control Power",
    )
    axis.plot(
        1_000.0 * time,
        arrays["baseline_contact_power_w"],
        color=COLORS["force"],
        linewidth=1.8,
        label="Two-Hand Contact Power",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Power Is Distinct from Couple Sign",
        xlabel="Time (ms)",
        ylabel="Power (W)",
    )
    axis.legend(frameon=False)

    axis = axes[1, 1]
    resultant = np.linalg.norm(arrays["baseline_resultant_contact_force_n"], axis=1)
    differential = np.linalg.norm(
        arrays["baseline_differential_contact_force_n"], axis=1
    )
    axis.plot(1_000.0 * time, resultant, color=COLORS["right"], label="Resultant Mode")
    axis.plot(
        1_000.0 * time,
        differential,
        color=COLORS["left"],
        label="Differential Mode",
    )
    axis.set(
        title="Common and Differential Force Modes",
        xlabel="Time (ms)",
        ylabel="Force Magnitude (N)",
    )
    axis.legend(frameon=False)
    figure.suptitle("Forward Two-Hand Interaction-Force Mechanism", fontweight="bold")
    return figure


def _draw_snapshot(
    axis: plt.Axes,
    state: FloatArray,
    forces: FloatArray,
    params: TwoArmParams,
    *,
    time_s: float,
    force_scale_m_n: float,
) -> None:
    points = kinematics(state, params)
    for side, color in (("right", COLORS["right"]), ("left", COLORS["left"])):
        chain = np.vstack(
            (
                points[f"{side}_shoulder"],
                points[f"{side}_elbow"],
                points[f"{side}_hand"],
            )
        )
        axis.plot(chain[:, 0], chain[:, 1], "o-", color=color, linewidth=2.1)
    direction = np.array([np.sin(state[6]), -np.cos(state[6])])
    center = points["club_center"]
    club = np.vstack((center - 0.25 * direction, center + 0.75 * direction))
    axis.plot(club[:, 0], club[:, 1], color="#111827", linewidth=3.0)
    for index, side in enumerate(("right", "left")):
        grip = points[f"{side}_grip"]
        delta = force_scale_m_n * forces[index]
        axis.annotate(
            "",
            xy=tuple((grip + delta).tolist()),
            xytext=tuple(grip.tolist()),
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS[side],
                "linewidth": 2.0,
            },
        )
    axis.scatter(float(center[0]), float(center[1]), s=22, color="black", zorder=4)
    axis.set(
        title=f"{1_000.0 * time_s:.0f} ms",
        xlim=(-0.72, 0.72),
        ylim=(-1.02, 0.16),
        xlabel="Target-Line Coordinate (m)",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)


def _force_geometry_figure(
    record: dict[str, Any], arrays: dict[str, FloatArray]
) -> plt.Figure:
    params = _params(record)
    time = arrays["baseline_time_s"]
    snapshot_times = (0.18, 0.20, 0.24, 0.30)
    indices = [int(np.argmin(np.abs(time - value))) for value in snapshot_times]
    forces = arrays["baseline_contact_force_on_club_n"][indices]
    maximum_force = float(np.max(np.linalg.norm(forces, axis=2)))
    force_scale = 0.18 / maximum_force
    figure, axes = plt.subplots(
        1, 4, figsize=(13.0, 4.25), sharey=True, constrained_layout=True
    )
    for axis, index in zip(axes, indices, strict=True):
        _draw_snapshot(
            axis,
            arrays["baseline_q"][index],
            arrays["baseline_contact_force_on_club_n"][index],
            params,
            time_s=float(time[index]),
            force_scale_m_n=force_scale,
        )
    axes[0].set_ylabel("Vertical Coordinate (m)")
    figure.suptitle(
        "Two-Hand Force Vectors Through the Couple Reversal",
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        (
            f"Arrow scale is common to all panels: longest arrow = "
            f"{maximum_force:.1f} N. Forces are hands on club."
        ),
        ha="center",
        fontsize=9,
    )
    return figure


def _numerical_audit_figure(
    record: dict[str, Any], arrays: dict[str, FloatArray]
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    rows = record["timestep_convergence"]
    step_ms = np.array([row["step_s"] for row in rows]) * 1_000.0
    residual = np.array([row["work_energy_residual_abs_j"] for row in rows])
    correction = np.array([row["projection_correction_max_m"] for row in rows])

    axis = axes[0, 0]
    axis.loglog(step_ms, residual, "o-", color=COLORS["energy"])
    axis.invert_xaxis()
    axis.set(
        title="Work-Energy Residual Converges",
        xlabel="Integrator Step (ms; Finer to the Right)",
        ylabel="Absolute Residual (J)",
    )
    axis.grid(which="both", alpha=0.25)

    axis = axes[0, 1]
    axis.loglog(step_ms, correction * 1e6, "o-", color=COLORS["branch"])
    axis.invert_xaxis()
    axis.set(
        title="Projection Correction Contracts",
        xlabel="Integrator Step (ms; Finer to the Right)",
        ylabel="Maximum Correction (µm)",
    )
    axis.grid(which="both", alpha=0.25)

    time = arrays["baseline_time_s"]
    energy_change = (
        arrays["baseline_mechanical_energy_j"]
        - arrays["baseline_mechanical_energy_j"][0]
    )
    control_work = np.zeros_like(time)
    control_power = arrays["baseline_control_power_w"]
    control_work[1:] = np.cumsum(
        0.5 * np.diff(time) * (control_power[:-1] + control_power[1:])
    )
    axis = axes[1, 0]
    axis.plot(
        1_000.0 * time,
        energy_change,
        color=COLORS["energy"],
        label="Mechanical-Energy Change",
    )
    axis.plot(
        1_000.0 * time,
        control_work,
        color=COLORS["work"],
        linestyle="--",
        label="Integrated Control Work",
    )
    axis.set(
        title="Physical Work-Energy Closure", xlabel="Time (ms)", ylabel="Energy (J)"
    )
    axis.legend(frameon=False)

    axis = axes[1, 1]
    axis.semilogy(
        1_000.0 * time,
        np.maximum(arrays["baseline_position_constraint_norm_m"], 1e-16),
        color=COLORS["neutral"],
    )
    axis.axhline(1e-8, color=COLORS["left"], linestyle="--", label="Declared Limit")
    axis.set(
        title="Closed-Loop Position Constraint",
        xlabel="Time (ms)",
        ylabel="Residual Norm (m)",
    )
    axis.legend(frameon=False)
    figure.suptitle("Forward Solver Closure and Sensitivity Audit", fontweight="bold")
    return figure


def render_figures(
    *,
    data_dir: str | Path = DATA_DIR,
    figure_dir: str | Path = FIGURE_DIR,
) -> tuple[Path, ...]:
    """Render three paired PDF/SVG figures from committed evidence."""
    source = Path(data_dir)
    destination = Path(figure_dir)
    record, arrays = _load_study(source)
    matplotlib.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9.5,
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "pdf.use14corefonts": True,
            "pdf.compression": 9,
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
            "svg.hashsalt": "forward-two-arm-phase-1",
        }
    )
    figures = (
        _couple_killswitch_figure(record, arrays),
        _force_geometry_figure(record, arrays),
        _numerical_audit_figure(record, arrays),
    )
    paths: list[Path] = []
    for stem, figure in zip(FIGURE_STEMS, figures, strict=True):
        paths.extend(_save(figure, destination, stem))
    return tuple(paths)


if __name__ == "__main__":
    render_figures()


__all__ = ["FIGURE_STEMS", "render_figures"]
