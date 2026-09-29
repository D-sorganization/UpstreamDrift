"""Create the publication figure for the joint work/load matching screen."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "joint-matched-proximal-rate-v1"
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE_ROOT = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
DATA_PATH = ARTICLE_ROOT / "data/joint_matched_proximal_rate_study.json"
FIGURE_DIR = ARTICLE_ROOT / "data/joint_matched_proximal_rate/figures"


def _save(figure: plt.Figure) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    name = "fig_joint_matched_proximal_rate"
    figure.savefig(
        FIGURE_DIR / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    svg_path = FIGURE_DIR / f"{name}.svg"
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={"Date": None},
    )
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def make_figure(record: dict) -> None:
    """Plot independent paired outcomes and tolerance sensitivity."""
    pairs = record["primary_match"]["pairs"]
    differences = np.asarray(
        [pair["impact_speed_difference_higher_minus_lower_m_s"] for pair in pairs]
    )
    order = np.argsort(differences)
    sensitivity = record["tolerance_sensitivity"]
    work = sorted({row["work_tolerance"] for row in sensitivity})
    load = sorted({row["load_tolerance"] for row in sensitivity})
    mean = np.full((len(work), len(load)), np.nan)
    counts = np.zeros((len(work), len(load)), dtype=int)
    faster = np.zeros_like(counts)
    for row in sensitivity:
        i = work.index(row["work_tolerance"])
        j = load.index(row["load_tolerance"])
        mean[i, j] = row["mean_impact_speed_difference_m_s"]
        counts[i, j] = row["independent_pair_count"]
        faster[i, j] = row["higher_rate_faster_pair_count"]

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    colors = np.where(differences[order] >= 0.0, "#2166ac", "#b2182b")
    axes[0].bar(np.arange(len(pairs)), differences[order], color=colors, width=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Independent Matched Pair (Sorted)")
    axes[0].set_ylabel("Higher-Rate Minus Lower-Rate Speed (m/s)")
    axes[0].set_title("Primary Work/Load Match Retains Both Signs")
    axes[0].grid(axis="y", alpha=0.25)

    image = axes[1].imshow(mean, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    axes[1].set_xticks(range(len(load)), [f"{100 * x:.0f}%" for x in load])
    axes[1].set_yticks(range(len(work)), [f"{100 * x:.1f}%" for x in work])
    axes[1].set_xlabel("Peak-Force Tolerance")
    axes[1].set_ylabel("Net and Positive Work Tolerance")
    axes[1].set_title("Mean Difference and Sign Count by Tolerance")
    for i in range(len(work)):
        for j in range(len(load)):
            axes[1].text(
                j,
                i,
                f"{mean[i, j]:+.2f}\n{faster[i, j]}/{counts[i, j]}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.colorbar(image, ax=axes[1], label="Mean Speed Difference (m/s)")
    figure.text(
        0.75,
        -0.02,
        "Cell count: higher-rate faster / independent pairs",
        ha="center",
        fontsize=8,
    )
    _save(figure)


def main() -> None:
    """Load committed evidence and generate the figure."""
    make_figure(json.loads(DATA_PATH.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
