"""Render the reviewer-facing event-topology robustness figure."""

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
REPORT_PATH = ARTICLE / "data/event_topology_channel_matrix.json"
FIGURE_STEM = ARTICLE / "figures/fig_event_topology_robustness"

CHANNEL_ORDER = ("both", "shoulder_only", "wrist_only", "zero")
CHANNEL_LABELS = ("Both", "Shoulder Only", "Wrist Only", "Zero Authority")
CHANNEL_COLORS = {
    "both": "#3a0ca3",
    "shoulder_only": "#e76f51",
    "wrist_only": "#2a9d8f",
    "zero": "#6c757d",
}
STATUS_COLORS = ("#d1495b", "#2a9d8f", "#f4a261", "#8338ec", "#4361ee")


def _channel_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["channel"]: record for record in report["channel_maps"]}


def _status_code(status: str) -> int:
    return {
        "absent": 0,
        "unique_transverse": 1,
        "multiple": 2,
        "grazing": 3,
        "numerical_failure": 4,
    }.get(status, 4)


def _status_label(status: str) -> str:
    return {
        "absent": "A",
        "unique_transverse": "U",
        "multiple": "M",
        "grazing": "G",
        "numerical_failure": "N",
    }.get(status, "N")


def _plot_nominal_topology(axis: plt.Axes, report: dict[str, Any]) -> None:
    indexed = _channel_index(report)
    delays_ms = np.asarray(report["registration"]["delays_s"]) * 1000.0
    statuses = [
        [outcome["status"] for outcome in indexed[channel]["nominal_by_delay"]]
        for channel in CHANNEL_ORDER
    ]
    matrix = np.asarray(
        [[_status_code(status) for status in row] for row in statuses], dtype=int
    )
    axis.imshow(
        matrix,
        cmap=ListedColormap(STATUS_COLORS),
        vmin=0,
        vmax=len(STATUS_COLORS) - 1,
        aspect="auto",
    )
    for row, values in enumerate(statuses):
        for column, status in enumerate(values):
            axis.text(
                column,
                row,
                _status_label(status),
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
            )
    axis.set_xticks(np.arange(len(delays_ms)), [f"{value:.0f}" for value in delays_ms])
    axis.set_yticks(np.arange(len(CHANNEL_LABELS)), CHANNEL_LABELS)
    axis.set_xlabel("Command Delay [ms]")
    axis.set_title("Nominal Global Topology")
    axis.legend(
        handles=(
            Patch(facecolor=STATUS_COLORS[1], label="U: Unique Positive Crossing"),
            Patch(facecolor=STATUS_COLORS[0], label="A: No Crossing"),
        ),
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.20),
    )


def _plot_preservation(axis: plt.Axes, report: dict[str, Any]) -> None:
    indexed = _channel_index(report)
    delays_ms = np.asarray(report["registration"]["delays_s"]) * 1000.0
    for channel, label in zip(CHANNEL_ORDER, CHANNEL_LABELS, strict=True):
        summaries = indexed[channel]["delay_summaries"]
        values = np.asarray([item["preservation_fraction"] for item in summaries])
        intervals = np.asarray([item["preservation_interval"] for item in summaries])
        lower = values - intervals[:, 0]
        upper = intervals[:, 1] - values
        axis.errorbar(
            delays_ms,
            values,
            yerr=np.vstack((lower, upper)),
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            capsize=2,
            color=CHANNEL_COLORS[channel],
            label=label,
        )
    axis.set_ylim(-0.03, 1.06)
    axis.set_xlabel("Command Delay [ms]")
    axis.set_ylabel("Topology-Preserved Pair Fraction")
    axis.set_title("Matched 1% Synthetic Perturbations")
    axis.legend(frameon=False, fontsize=7.5, ncol=2)
    axis.text(
        0.02,
        0.04,
        "Preserved absence is not crossing success.",
        transform=axis.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#adb5bd", "pad": 2.5},
    )


def _plot_step_controls(axis: plt.Axes, report: dict[str, Any]) -> None:
    summaries = {item["channel"]: item for item in report["step_refinement_summary"]}
    time_ns = np.asarray(
        [
            1.0e9 * summaries[channel]["maximum_event_time_residual_s"]
            for channel in CHANNEL_ORDER
        ]
    )
    axis.bar(
        np.arange(len(CHANNEL_ORDER)),
        time_ns,
        color=[CHANNEL_COLORS[channel] for channel in CHANNEL_ORDER],
        width=0.65,
    )
    axis.set_xticks(np.arange(len(CHANNEL_LABELS)), CHANNEL_LABELS, rotation=15)
    axis.set_ylabel("Maximum Event-Time Residual [ns]")
    axis.set_title("Integration-Step Refinement")
    axis.text(
        0.03,
        0.95,
        "0.001 / 0.002 / 0.004 s:\nall topology identities match",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#2a9d8f", "pad": 3.0},
    )


def _plot_horizon_controls(axis: plt.Axes, report: dict[str, Any]) -> None:
    records = {
        (item["channel"], item["horizon_s"]): item
        for item in report["horizon_controls"]
    }
    horizons = report["registration"]["horizons_s"]
    statuses = [
        [records[(channel, horizon)]["status"] for horizon in horizons]
        for channel in CHANNEL_ORDER
    ]
    matrix = np.asarray(
        [[_status_code(status) for status in row] for row in statuses], dtype=int
    )
    axis.imshow(
        matrix,
        cmap=ListedColormap(STATUS_COLORS),
        vmin=0,
        vmax=len(STATUS_COLORS) - 1,
        aspect="auto",
    )
    for row, values in enumerate(statuses):
        for column, status in enumerate(values):
            axis.text(
                column,
                row,
                _status_label(status),
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
            )
    axis.set_xticks(np.arange(len(horizons)), [f"{value:.2f}" for value in horizons])
    axis.set_yticks(np.arange(len(CHANNEL_LABELS)), CHANNEL_LABELS)
    axis.set_xlabel("Global Search Horizon [s]")
    axis.set_title("Horizon Control Identifies Wrist-Only Truncation")


def _normalize_svg(svg_path: Path) -> None:
    """Remove serializer-only trailing whitespace from deterministic SVG output."""

    lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    """Render topology, perturbation, refinement, and horizon controls."""

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
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    _plot_nominal_topology(axes[0, 0], report)
    _plot_preservation(axes[0, 1], report)
    _plot_step_controls(axes[1, 0], report)
    _plot_horizon_controls(axes[1, 1], report)
    figure.suptitle(
        "Channel Masks Expose Topology Loss and Horizon Truncation",
        fontsize=11,
    )
    figure.savefig(
        FIGURE_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Open Research"},
    )
    svg_path = FIGURE_STEM.with_suffix(".svg")
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "Open Research"},
    )
    _normalize_svg(svg_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
