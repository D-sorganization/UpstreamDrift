"""Render the stateful distributed-grip falsification figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_stateful_distributed_smoke/summary.json"
OUTPUT = ARTICLE / "figures/fig_stateful_distributed_grip_falsification"

VARIANT_LABELS = {
    "nominal": "Nominal",
    "frictionless_killswitch": "Frictionless Killswitch",
    "low_friction_slip_probe": "Low-Friction Slip Probe",
    "high_friction": "High Friction",
    "low_tangential_stiffness": "Low Tangential Stiffness",
    "high_tangential_stiffness": "High Tangential Stiffness",
    "zero_preload": "Zero Preload",
    "velocity_reversed": "Velocity Reversed",
    "opening_probe": "Opening Probe",
}


def load_summary(path: Path = DATA) -> dict[str, Any]:
    """Load and minimally validate a published stateful summary."""

    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "stateful-distributed-forward-summary/v1":
        raise ValueError("unsupported stateful distributed summary schema")
    if not isinstance(summary.get("groups"), list):
        raise ValueError("stateful distributed summary must contain groups")
    return summary


def completed_groups(
    summary: dict[str, Any], *, passed: bool | None = None
) -> list[dict[str, Any]]:
    """Return completed groups, optionally filtered by registered pass state."""

    groups = [
        group for group in summary["groups"] if group.get("status") == "completed"
    ]
    if passed is not None:
        groups = [group for group in groups if group.get("passes") is passed]
    return groups


def _line_style(group: dict[str, Any]) -> dict[str, Any]:
    if group["variant"] == "nominal":
        return {"color": "#1f4e79", "linewidth": 2.2, "alpha": 1.0}
    if not group["passes"]:
        return {"color": "#b22222", "linewidth": 2.0, "alpha": 1.0}
    return {"color": "#7f8c8d", "linewidth": 0.9, "alpha": 0.55}


def _residual_panel(
    axis: plt.Axes,
    groups: list[dict[str, Any]],
    key: str,
    ylabel: str,
) -> None:
    labeled: set[str] = set()
    for group in groups:
        style = _line_style(group)
        category = (
            "Nominal"
            if group["variant"] == "nominal"
            else "Registered Failure"
            if not group["passes"]
            else "Other Passing Variants"
        )
        label = category if category not in labeled else None
        labeled.add(category)
        axis.semilogy(
            np.asarray(group["time_steps_s"]) * 1_000.0,
            group[key],
            marker="o",
            markersize=3.5,
            label=label,
            **style,
        )
    axis.invert_xaxis()
    axis.set_xlabel("Time Step (ms; Finer to the Right)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)


def _regime_panel(axis: plt.Axes, groups: list[dict[str, Any]]) -> None:
    variants = [group["variant"] for group in groups]
    regimes = ("elastic_stick", "coulomb_slip", "open")
    colors = ("#2f855a", "#d97706", "#64748b")
    left = np.zeros(len(groups), dtype=float)
    for regime, color in zip(regimes, colors, strict=True):
        values = np.asarray(
            [group["fine_step_regime_counts"].get(regime, 0) for group in groups],
            dtype=float,
        )
        totals = np.asarray(
            [sum(group["fine_step_regime_counts"].values()) for group in groups],
            dtype=float,
        )
        fractions = np.divide(
            values, totals, out=np.zeros_like(values), where=totals > 0
        )
        axis.barh(
            np.arange(len(groups)),
            fractions,
            left=left,
            color=color,
            label=VARIANT_LABELS[regime]
            if regime in VARIANT_LABELS
            else regime.replace("_", " ").title(),
        )
        left += fractions
    axis.set_yticks(np.arange(len(groups)), [VARIANT_LABELS[item] for item in variants])
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Fraction of Fine-Step Station Intervals")
    axis.grid(axis="x", alpha=0.25)


def _speed_panel(axis: plt.Axes, summary: dict[str, Any]) -> None:
    rows = summary["counterfactuals"]
    labels = [VARIANT_LABELS[row["variant"]] for row in rows]
    values = np.asarray([row["clubhead_speed_difference_m_s"] for row in rows])
    colors = ["#b22222" if value < 0 else "#2f855a" for value in values]
    axis.barh(np.arange(len(rows)), values, color=colors)
    axis.set_yticks(np.arange(len(rows)), labels)
    axis.invert_yaxis()
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Fine-Step Speed Difference From Nominal (m/s)")
    axis.grid(axis="x", alpha=0.25)
    axis.text(
        0.02,
        0.03,
        "Separately integrated 5 ms engineering counterfactuals;\nnot matched delivery or human evidence.",
        transform=axis.transAxes,
        fontsize=7,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )


def render_figure(summary: dict[str, Any], output: Path = OUTPUT) -> None:
    """Render deterministic PDF and SVG evidence views."""

    groups = completed_groups(summary)
    if len(groups) != 9:
        raise ValueError("expected nine completed MuJoCo stateful groups")
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.7), constrained_layout=True)
    _residual_panel(
        axes[0, 0],
        groups,
        "trajectory_energy_relative_residuals",
        "Passive Energy Relative Defect",
    )
    axes[0, 0].set_title("A. Passive Energy Defect Contracts")
    _residual_panel(
        axes[0, 1],
        groups,
        "coupling_work_relative_residuals",
        "Coupling-Work Relative Defect",
    )
    axes[0, 1].set_title("B. Two Variants Fail the Frozen Refinement Gate")
    _regime_panel(axes[1, 0], groups)
    axes[1, 0].set_title("C. Fine-Step Contact-Regime Composition")
    _speed_panel(axes[1, 1], summary)
    axes[1, 1].set_title("D. Short-Horizon Counterfactual Sensitivity")
    axes[0, 0].legend(fontsize=7, loc="best")
    axes[0, 1].legend(fontsize=7, loc="best")
    axes[1, 0].legend(fontsize=7, loc="lower right")
    figure.suptitle(
        "Stateful Distributed-Grip Falsification\n"
        "27 MuJoCo Cases Complete; 27 Pinocchio Cases Unavailable; Promotion Withheld",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = output.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def main() -> None:
    render_figure(load_summary())


if __name__ == "__main__":
    main()
