"""Render publication figures for coupled base and club compliance."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.moving_base_flexible_club import (
    MovingBaseFlexibleParams,
    kinematics,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
FIGURE_DIR = OUTPUT_ROOT / "figures"
FIGURE_STEMS = (
    "fig_coupled_base_flex_force_geometry",
    "fig_coupled_base_flex_transfer",
    "fig_coupled_base_flex_falsification",
)

INK = "#263238"
BLUE = "#246A8D"
ORANGE = "#E07A2D"
RED = "#B23A48"
GREEN = "#2A7F62"
PURPLE = "#7251A3"
GRAY = "#7A858A"


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads(
        (DATA_DIR / "moving_base_flexible_study.json").read_text(encoding="utf-8")
    )
    with np.load(DATA_DIR / "moving_base_flexible_study.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    return record, arrays


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(
            path,
            format=suffix,
            metadata={"Creator": "UpstreamDrift Reproducible Research"},
        )
        outputs.append(path)
    plt.close(fig)
    return outputs


def _force_geometry(arrays: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    params = MovingBaseFlexibleParams.publication_default()
    time = arrays["baseline_time_s"]
    q = arrays["baseline_q"]
    contacts = arrays["baseline_contact_force_on_club_n"]
    couple = arrays["baseline_force_generated_couple_nm"]
    targets = (0.10, 0.22, 0.30, 0.40)
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.25), sharex=True, sharey=True)
    scale = 0.0025
    for axis, target in zip(axes, targets, strict=True):
        index = int(np.argmin(np.abs(time - target)))
        points = kinematics(q[index], params)
        axis.plot(
            [
                points["right_shoulder"][0],
                points["right_elbow"][0],
                points["right_hand"][0],
            ],
            [
                points["right_shoulder"][1],
                points["right_elbow"][1],
                points["right_hand"][1],
            ],
            "o-",
            color=BLUE,
            lw=2,
            ms=3,
        )
        axis.plot(
            [
                points["left_shoulder"][0],
                points["left_elbow"][0],
                points["left_hand"][0],
            ],
            [
                points["left_shoulder"][1],
                points["left_elbow"][1],
                points["left_hand"][1],
            ],
            "o-",
            color=ORANGE,
            lw=2,
            ms=3,
        )
        axis.plot(
            [points["grip_center"][0], points["flex_joint"][0]],
            [points["grip_center"][1], points["flex_joint"][1]],
            color=INK,
            lw=3,
        )
        axis.plot(
            [points["flex_joint"][0], points["clubhead"][0]],
            [points["flex_joint"][1], points["clubhead"][1]],
            color=PURPLE,
            lw=3,
        )
        axis.scatter(*points["base"], marker="s", s=28, color=GRAY, zorder=5)
        for point_name, force, color, label in (
            ("right_grip", contacts[index, 0], RED, "R"),
            ("left_grip", contacts[index, 1], GREEN, "L"),
        ):
            point = points[point_name]
            delta = scale * force
            axis.arrow(
                point[0],
                point[1],
                delta[0],
                delta[1],
                width=0.006,
                head_width=0.045,
                length_includes_head=True,
                color=color,
                alpha=0.95,
                zorder=6,
            )
            axis.text(
                point[0] + delta[0],
                point[1] + delta[1],
                label,
                color=color,
                fontsize=7,
                fontweight="bold",
            )
        sign = "Negative" if couple[index] < 0.0 else "Positive"
        axis.set_title(
            f"$t$ = {time[index]:.2f} s\n{sign} Couple: {couple[index]:.1f} N m"
        )
        axis.axhline(0.0, color="#D6DADD", lw=0.6)
        axis.axvline(0.0, color="#D6DADD", lw=0.6)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-0.85, 1.1)
        axis.set_ylim(-1.35, 0.55)
        axis.set_xlabel("Target Direction [m]")
    axes[0].set_ylabel("Vertical Position [m]")
    fig.suptitle(
        "Endogenous Base Motion, Club Flex, and Two-Hand Interaction Forces",
        y=1.02,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.01, 1.0, 0.98))
    return _save(fig, output_dir, FIGURE_STEMS[0])


def _transfer(arrays: dict[str, np.ndarray], output_dir: Path) -> list[Path]:
    time = arrays["baseline_time_s"]
    q = arrays["baseline_q"]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.1), sharex=True)
    axes[0, 0].plot(time, 1000.0 * q[:, 4], color=BLUE, label="Horizontal")
    axes[0, 0].plot(time, 1000.0 * q[:, 5], color=ORANGE, label="Vertical")
    axes[0, 0].set_ylabel("Base Displacement [mm]")
    axes[0, 0].set_title("Finite-Mass Base Response")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        time,
        arrays["baseline_force_generated_couple_nm"],
        color=RED,
        label="Force-Generated",
    )
    axes[0, 1].plot(
        time,
        arrays["baseline_direct_wrist_torque_nm"],
        color=INK,
        ls="--",
        label="Direct Wrist",
    )
    axes[0, 1].axhline(0.0, color=GRAY, lw=0.7)
    axes[0, 1].set_ylabel("Grip Couple [N m]")
    axes[0, 1].set_title("Separated Interaction and Command")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(time, np.rad2deg(q[:, 9]), color=PURPLE, label="Flex")
    moment_axis = axes[1, 0].twinx()
    moment_axis.plot(
        time,
        arrays["baseline_shaft_elastic_moment_nm"],
        color=RED,
        label="Elastic Moment",
    )
    moment_axis.plot(
        time,
        arrays["baseline_shaft_damping_moment_nm"],
        color=ORANGE,
        ls="--",
        label="Damping Moment",
    )
    axes[1, 0].set_ylabel("Shaft Flex [deg]", color=PURPLE)
    moment_axis.set_ylabel("Internal Moment [N m]")
    axes[1, 0].set_title("Compliant-Club Storage and Release")
    lines = axes[1, 0].lines + moment_axis.lines
    axes[1, 0].legend(lines, [line.get_label() for line in lines], frameon=False)

    axes[1, 1].plot(
        time,
        arrays["baseline_shaft_strain_energy_j"],
        color=PURPLE,
        label="Shaft Strain Energy",
    )
    speed_axis = axes[1, 1].twinx()
    speed_axis.plot(
        time,
        np.linalg.norm(arrays["baseline_clubhead_velocity_m_s"], axis=1),
        color=GREEN,
        label="Clubhead Speed",
    )
    axes[1, 1].set_ylabel("Shaft Strain Energy [J]", color=PURPLE)
    speed_axis.set_ylabel("Clubhead Speed [m/s]", color=GREEN)
    axes[1, 1].set_title("Distal Response")
    distal_lines = axes[1, 1].lines + speed_axis.lines
    axes[1, 1].legend(
        distal_lines,
        [line.get_label() for line in distal_lines],
        frameon=False,
    )
    for axis in axes[1]:
        axis.set_xlabel("Time [s]")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Coupled Transfer Observables", fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_dir, FIGURE_STEMS[1])


def _falsification(
    record: dict, arrays: dict[str, np.ndarray], output_dir: Path
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    time = arrays["baseline_time_s"]
    mask = (time >= 0.18) & (time <= 0.25)
    axes[0].plot(
        time[mask],
        arrays["baseline_force_generated_couple_nm"][mask],
        color=RED,
        label="Continued Command",
    )
    axes[0].plot(
        arrays["branch_time_s"],
        arrays["branch_force_generated_couple_nm"],
        color=BLUE,
        ls="--",
        label="Zero Command at 0.20 s",
    )
    axes[0].axhline(0.0, color=GRAY, lw=0.7)
    axes[0].set_title("Same-State Intervention")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Force-Generated Couple [N m]")
    axes[0].legend(frameon=False)

    rows = record["mechanism_sensitivity"]
    labels = [
        "Base\n12 kN/m",
        "Base\n48 kN/m",
        "Shaft\n40 N m/rad",
        "Shaft\n160 N m/rad",
        "No Shaft\nDamping",
    ]
    minima = [row["minimum_force_generated_couple_nm"] for row in rows]
    axes[1].bar(range(len(rows)), minima, color=[BLUE, BLUE, PURPLE, PURPLE, ORANGE])
    axes[1].set_xticks(range(len(rows)), labels, rotation=20, ha="right")
    axes[1].axhline(0.0, color=GRAY, lw=0.7)
    axes[1].set_ylabel("Minimum Couple [N m]")
    axes[1].set_title("Mechanism Sensitivity")

    convergence = record["timestep_convergence"]
    step = np.array([row["step_s"] for row in convergence])
    residual = np.array([row["work_energy_residual_abs_j"] for row in convergence])
    projection = np.array([row["projection_correction_max_m"] for row in convergence])
    axes[2].loglog(
        step * 1000.0, residual, "o-", color=GREEN, label="Energy Residual [J]"
    )
    axes[2].loglog(
        step * 1000.0,
        projection * 1e6,
        "s--",
        color=PURPLE,
        label="Projection [µm]",
    )
    axes[2].invert_xaxis()
    axes[2].set_xlabel("Timestep [ms]")
    axes[2].set_title("Resolution Audit")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Counterfactual and Numerical Falsification Tests", fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_dir, FIGURE_STEMS[2])


def render_figures(output_dir: Path = FIGURE_DIR) -> list[Path]:
    """Render every figure as vector PDF and SVG."""
    _style()
    record, arrays = _load()
    return [
        *_force_geometry(arrays, output_dir),
        *_transfer(arrays, output_dir),
        *_falsification(record, arrays, output_dir),
    ]


def main() -> None:
    render_figures()


if __name__ == "__main__":
    main()
