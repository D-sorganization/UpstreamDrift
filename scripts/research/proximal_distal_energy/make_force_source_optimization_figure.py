"""Render the Coriolis-impulse optimization comparison from committed JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
DATA_PATH = ARTICLE / "data/force_source_optimization.json"
FIGURE_STEM = ARTICLE / "figures/fig_force_source_optimization"


def _matching_outcome(outcomes: list[dict], candidate: dict) -> dict:
    for outcome in outcomes:
        if outcome["candidate"] == {
            key: candidate[key]
            for key in (
                "shoulder_torque_nm",
                "wrist_drive_nm",
                "wrist_restrain_nm",
                "onset_s",
            )
        }:
            return outcome
    raise ValueError("summary candidate is absent from outcome rows")


def main() -> None:
    """Render the objective tradeoff and component-work comparison."""
    artifact = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    qualified = [
        row for row in artifact["outcomes"] if row["status"] == "qualified_impact"
    ]
    summary = artifact["summary"]
    impulse = summary["maximum_coriolis_impulse"]
    speed = summary["maximum_clubhead_speed"]
    impulse_row = _matching_outcome(qualified, impulse)
    speed_row = _matching_outcome(qualified, speed)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    scatter = axes[0].scatter(
        [row["coriolis_absolute_tangent_impulse_n_s"] for row in qualified],
        [row["clubhead_speed_m_s"] for row in qualified],
        c=[row["candidate"]["onset_s"] for row in qualified],
        cmap="viridis",
        s=32,
        alpha=0.76,
        edgecolors="none",
    )
    axes[0].scatter(
        impulse["coriolis_absolute_tangent_impulse_n_s"],
        impulse["clubhead_speed_m_s"],
        marker="*",
        s=190,
        color="#dc2626",
        label="Max absolute Coriolis impulse",
        zorder=5,
    )
    axes[0].scatter(
        speed["coriolis_absolute_tangent_impulse_n_s"],
        speed["clubhead_speed_m_s"],
        marker="D",
        s=72,
        color="#111827",
        label="Max clubhead speed",
        zorder=5,
    )
    axes[0].set(
        xlabel="Absolute Coriolis Tangent Impulse [N s]",
        ylabel="Impact Clubhead Speed [m/s]",
        title="Registered Candidates: Different Objectives",
    )
    axes[0].legend(fontsize=8)
    fig.colorbar(scatter, ax=axes[0], label="Wrist-drive onset [s]")

    components = ("coriolis", "squared_speed", "gravity", "applied")
    labels = (
        "Coriolis\ncross-speed",
        "Squared-speed\ncentripetal",
        "Gravity",
        "Applied",
    )
    x = np.arange(len(components))
    width = 0.36
    for offset, row, label, color in (
        (-width / 2, impulse_row, "Max impulse", "#dc2626"),
        (width / 2, speed_row, "Max speed", "#2563eb"),
    ):
        axes[1].bar(
            x + offset,
            [row[f"{component}_work_j"] for component in components],
            width,
            label=label,
            color=color,
        )
    axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Generalized Work Through Impact [J]",
        title="Equation-Term Work Is Signed",
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Coordinate-Explicit Pendulum Source Attribution",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIGURE_STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
