"""Render the reviewer-facing bounded event-reachability figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REPORT_PATH = ARTICLE / "data/bounded_event_reachability.json"
FIGURE_STEM = ARTICLE / "figures/fig_bounded_event_reachability"

CHANNEL_ORDER = ("both", "shoulder_only", "wrist_only", "zero")
CHANNEL_LABELS = ("Both", "Shoulder Only", "Wrist Only", "Zero Authority")
TARGET_ORDER = (
    "minus_0p002",
    "minus_0p001",
    "minus_0p0005",
    "zero",
    "plus_0p0005",
    "plus_0p001",
    "plus_0p002",
)
TARGET_LABELS = ("-2.0", "-1.0", "-0.5", "0.0", "+0.5", "+1.0", "+2.0")
CHANNEL_COLORS = {
    "both": "#3a0ca3",
    "shoulder_only": "#e76f51",
    "wrist_only": "#2a9d8f",
    "zero": "#6c757d",
}


def _trial_index(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (record["target_name"], record["channel"]): record
        for record in report["continuation_trials"]
    }


def _plot_feasibility(axis: plt.Axes, report: dict[str, Any]) -> None:
    indexed = _trial_index(report)
    matrix = np.asarray(
        [
            [
                indexed[(target, channel)]["replay_feasibility_status"] == "feasible"
                for target in TARGET_ORDER
            ]
            for channel in CHANNEL_ORDER
        ],
        dtype=int,
    )
    axis.imshow(
        matrix,
        cmap=ListedColormap(("#d1495b", "#2a9d8f")),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            label = "Pass" if matrix[row, column] else "Fail"
            axis.text(column, row, label, ha="center", va="center", color="white")
    axis.set_xticks(np.arange(len(TARGET_LABELS)), TARGET_LABELS)
    axis.set_yticks(np.arange(len(CHANNEL_LABELS)), CHANNEL_LABELS)
    axis.set_xlabel("Event-Tangent Angle Offset [mrad]")
    axis.set_title("Registered Feasibility Matrix")
    axis.legend(
        handles=(
            Patch(facecolor="#2a9d8f", label="Feasible"),
            Patch(facecolor="#d1495b", label="Infeasible"),
        ),
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.18),
    )


def _plot_event_time(axis: plt.Axes, report: dict[str, Any]) -> None:
    indexed = _trial_index(report)
    target_offsets = np.asarray((-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0))
    nominal_time_ms = 1000.0 * indexed[("zero", "both")]["event_time_s"]
    for channel, label in zip(CHANNEL_ORDER, CHANNEL_LABELS, strict=True):
        records = [indexed[(target, channel)] for target in TARGET_ORDER]
        event_times = np.asarray(
            [
                np.nan
                if record["replay_feasibility_status"] != "feasible"
                else 1000.0 * record["event_time_s"] - nominal_time_ms
                for record in records
            ]
        )
        axis.plot(
            target_offsets,
            event_times,
            marker="o",
            color=CHANNEL_COLORS[channel],
            label=label,
        )
    axis.axhline(0.0, color="#adb5bd", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Event-Tangent Angle Offset [mrad]")
    axis.set_ylabel("Event-Time Shift [ms]")
    axis.set_title("Event Timing After Independent Replay")
    axis.legend(frameon=False, fontsize=7.5, ncol=2)


def _plot_refinement(axis: plt.Axes, report: dict[str, Any]) -> None:
    controls = report["falsification_controls"]
    mesh = controls["mesh_refinement"]
    steps = controls["integration_step_refinement"]
    base_objective = next(
        record["objective"] for record in mesh if record["segment_count"] == 4
    )
    mesh_x = np.asarray([record["segment_count"] for record in mesh])
    mesh_y = np.asarray([record["objective"] / base_objective for record in mesh])
    step_x = np.asarray([record["dt_s"] * 1000.0 for record in steps])
    step_y = np.asarray([record["objective"] / base_objective for record in steps])
    axis.plot(mesh_x, mesh_y, marker="o", color="#4361ee", label="Shooting Mesh")
    twin = axis.twiny()
    twin.plot(step_x, step_y, marker="s", color="#f4a261", label="RK4 Step")
    axis.axhline(1.0, color="#adb5bd", linewidth=0.8, linestyle="--")
    axis.set_xticks(mesh_x)
    axis.set_xlabel("Shooting Segment Count")
    twin.set_xticks(step_x)
    twin.set_xlabel("RK4 Step [ms]")
    axis.set_ylabel("Objective / Registered Objective")
    axis.set_title("Mesh and Integration-Step Controls")
    handles = axis.lines[:1] + twin.lines[:1]
    axis.legend(handles=handles, frameon=False, loc="best")


def _plot_multistart(axis: plt.Axes, report: dict[str, Any]) -> None:
    controls = report["falsification_controls"]["multistart"]
    objectives = np.asarray([record["objective"] for record in controls])
    relative = objectives / objectives.min()
    axis.bar(
        np.arange(len(controls)),
        relative,
        color=("#4361ee", "#f4a261"),
        width=0.6,
    )
    axis.axhline(1.05, color="#d1495b", linestyle="--", linewidth=1.1)
    axis.set_xticks(
        np.arange(len(controls)),
        [f"Seed {record['seed']}" for record in controls],
    )
    axis.set_ylabel("Objective / Best Objective")
    axis.set_title("Multistart Optimality Gate Fails")
    spread = report["qualification"]["multistart_relative_objective_spread"]
    axis.text(
        0.04,
        0.94,
        f"Observed Spread: {100.0 * spread:.2f}%\nRegistered Gate: 5.00%",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#d1495b", "pad": 3.0},
    )


def main() -> None:
    """Render feasibility, event timing, refinement, and optimality controls."""

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    _plot_feasibility(axes[0, 0], report)
    _plot_event_time(axes[0, 1], report)
    _plot_refinement(axes[1, 0], report)
    _plot_multistart(axes[1, 1], report)
    figure.suptitle(
        "Bounded Event Reachability Separates Feasibility From Optimality",
        fontsize=11,
    )
    figure.savefig(
        FIGURE_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Open Research"},
    )
    figure.savefig(
        FIGURE_STEM.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "Open Research"},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
