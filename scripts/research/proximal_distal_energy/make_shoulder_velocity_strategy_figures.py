"""Create publication figures for the trajectory-level strategy search."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "proximal-distal-shoulder-strategy-v3"
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_PATH = ARTICLE_ROOT / "data" / "shoulder_velocity_strategy_study.json"
FIGURE_DIR = ARTICLE_ROOT / "data" / "shoulder_velocity_transfer" / "figures"


def _save(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        FIGURE_DIR / f"{name}.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(figure)


def _valid_rows(record: dict) -> list[dict]:
    return [row for row in record["programs"] if row["valid_impact"]]


def make_association_figure(record: dict) -> None:
    """Plot release velocity against speed and braking exposure."""
    rows = _valid_rows(record)
    velocity = np.asarray([row["proximal_velocity_at_release_rad_s"] for row in rows])
    speed = np.asarray([row["impact_speed_m_s"] for row in rows])
    braking = np.asarray([row["braking_grip_work_j"] for row in rows])
    release = np.asarray([row["wrist_release_s"] for row in rows])
    actuator_work = np.asarray([row["total_actuator_work_j"] for row in rows])
    index_to_row = {row["program_index"]: row for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.1), constrained_layout=True)
    scatter = axes[0].scatter(velocity, speed, c=release, cmap="viridis", s=48)
    axes[0].set_xlabel("Proximal-Link Velocity at Release (rad/s)")
    axes[0].set_ylabel("Impact Speed (m/s)")
    axes[0].set_title("Higher Release Velocity Is Not a Standalone Benefit")
    figure.colorbar(scatter, ax=axes[0], label="Wrist Release Time (s)")
    axes[1].scatter(velocity, braking, c=release, cmap="viridis", s=48)
    axes[1].set_xlabel("Proximal-Link Velocity at Release (rad/s)")
    axes[1].set_ylabel("Negative Interface Work After Release (J)")
    axes[1].set_title("Later High-Velocity Releases Increase Braking Exposure")
    axes[2].scatter(actuator_work, speed, c=velocity, cmap="plasma", s=48)
    for pair in record["matched_work_screen"]["pairs"]:
        low = index_to_row[pair["lower_velocity_program_index"]]
        high = index_to_row[pair["higher_velocity_program_index"]]
        axes[2].plot(
            [low["total_actuator_work_j"], high["total_actuator_work_j"]],
            [low["impact_speed_m_s"], high["impact_speed_m_s"]],
            color="#6b7280",
            linewidth=0.8,
            alpha=0.55,
        )
    axes[2].set_xlabel("Net Actuator Work to Impact (J)")
    axes[2].set_ylabel("Impact Speed (m/s)")
    axes[2].set_title("Work-Matched Pairs Retain a Load Confound")
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(figure, "fig_shoulder_velocity_strategy_associations")


def make_pareto_figure(record: dict) -> None:
    """Plot the speed, braking-work, and peak-force tradeoff."""
    rows = _valid_rows(record)
    pareto = set(record["pareto_program_indices"])
    speed = np.asarray([row["impact_speed_m_s"] for row in rows])
    braking = np.asarray([row["braking_grip_work_j"] for row in rows])
    peak_force = np.asarray([row["peak_grip_force_n"] for row in rows])
    is_pareto = np.asarray([row["program_index"] in pareto for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    sizes = 30.0 + 100.0 * peak_force / np.max(peak_force)
    axis.scatter(
        braking[~is_pareto], speed[~is_pareto], s=sizes[~is_pareto], alpha=0.45
    )
    axis.scatter(
        braking[is_pareto],
        speed[is_pareto],
        s=sizes[is_pareto],
        facecolors="none",
        edgecolors="#b2182b",
        linewidths=1.5,
        label="Nondominated",
    )
    axis.set_xlabel("Negative Interface Work After Release (J; Minimize)")
    axis.set_ylabel("Impact Speed (m/s; Maximize)")
    axis.set_title("Speed, Braking, and Peak-Force Objectives Remain in Tension")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    axis.text(
        0.98,
        0.03,
        "Marker size scales with peak net interface force",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    _save(figure, "fig_shoulder_velocity_strategy_pareto")


def make_grid_figure(record: dict) -> None:
    """Plot impact speed across shoulder-cut and wrist-release timing."""
    cuts = record["grid"]["shoulder_cut_s"]
    releases = record["grid"]["wrist_release_s"]
    after_values = record["grid"]["shoulder_torque_after_nm"]
    rows = record["programs"]
    figure, axes = plt.subplots(
        1, len(after_values), figsize=(11.2, 3.8), constrained_layout=True
    )
    images = []
    for axis, after in zip(axes, after_values, strict=True):
        values = np.full((len(cuts), len(releases)), np.nan)
        for row in rows:
            if row["valid_impact"] and np.isclose(
                row["shoulder_torque_after_nm"], after
            ):
                i = cuts.index(row["shoulder_cut_s"])
                j = releases.index(row["wrist_release_s"])
                values[i, j] = row["impact_speed_m_s"]
        image = axis.imshow(values, origin="lower", aspect="auto", vmin=18.0, vmax=40.0)
        images.append(image)
        axis.set_xticks(range(len(releases)), [f"{value:.2f}" for value in releases])
        axis.set_yticks(range(len(cuts)), [f"{value:.2f}" for value in cuts])
        axis.set_xlabel("Wrist Release Time (s)")
        axis.set_title(f"Post-Cut Shoulder Torque: {after:.0f} N m")
        for i in range(len(cuts)):
            for j in range(len(releases)):
                label = "X" if np.isnan(values[i, j]) else f"{values[i, j]:.1f}"
                axis.text(j, i, label, ha="center", va="center", fontsize=8)
    axes[0].set_ylabel("Shoulder-Drive Cut Time (s)")
    figure.colorbar(images[-1], ax=axes, label="Impact Speed (m/s)", shrink=0.9)
    figure.suptitle("Only 26 of 60 Programs Reach the Registered Impact Window")
    _save(figure, "fig_shoulder_velocity_strategy_grid")


def main() -> None:
    """Load committed evidence and generate all strategy figures."""
    record = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    make_association_figure(record)
    make_pareto_figure(record)
    make_grid_figure(record)


if __name__ == "__main__":
    main()
