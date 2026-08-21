"""Render the inertia-and-bias transport validity-horizon summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
STEM = STUDY_DIR / "figures/fig_forward_contact_validity_horizon"


def main() -> int:
    """Write one vector-first four-panel adverse-load failure map."""

    record = json.loads(
        (STUDY_DIR / "data/forward_contact_validity_horizon.json").read_text(
            encoding="utf-8"
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 10))
    for color, variant in zip(colors, record["results"]["variants"], strict=True):
        rows = variant["by_horizon"]
        horizon_ms = np.asarray([row["horizon_s"] for row in rows]) * 1.0e3
        label = variant["variant_id"].replace("_", " ").title()
        axes[0, 0].plot(
            horizon_ms,
            [row["pass_fraction"] * 100.0 for row in rows],
            marker="o",
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            horizon_ms,
            [row["worst_position_max_m"] * 1.0e6 for row in rows],
            marker="o",
            color=color,
        )
        axes[1, 0].plot(
            horizon_ms,
            [row["worst_wrench_relative_rms"] * 100.0 for row in rows],
            marker="o",
            color=color,
        )
        axes[1, 1].plot(
            horizon_ms,
            [row["worst_energy_closure"] * 100.0 for row in rows],
            marker="o",
            color=color,
        )
    axes[0, 0].set(title="Complete-Gate Pass Fraction", ylabel="Cases Passing (%)")
    axes[0, 0].set_ylim(95.0, 100.5)
    axes[0, 1].set(
        title="Worst Club-Position Difference",
        ylabel="Maximum Difference (µm)",
        yscale="log",
    )
    axes[1, 0].set(
        title="Worst Contact-Wrench Difference",
        ylabel="Relative RMS (%)",
        yscale="log",
    )
    axes[1, 1].set(
        title="Worst Work-Energy Closure Residual",
        ylabel="Normalized Residual (%)",
        yscale="log",
    )
    axes[1, 1].axhline(8.0, color="black", linestyle="--", linewidth=1.0)
    for axis in axes.flat:
        axis.set_xlabel("Forward Horizon (ms)")
        axis.grid(alpha=0.25)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=5,
        fontsize=8,
    )
    fig.suptitle(
        "No Inertia-and-Bias Transport Failure Was Observed Through 50 ms",
        fontsize=14,
    )
    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        STEM.with_suffix(".pdf"),
        metadata={"Title": "Forward-Contact Validity Horizon"},
    )
    fig.savefig(
        STEM.with_suffix(".svg"),
        metadata={"Title": "Forward-Contact Validity Horizon"},
    )
    svg_path = STEM.with_suffix(".svg")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
