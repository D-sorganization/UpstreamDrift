"""Render the finite-ground mechanism and registered-atlas figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_ground_atlas.npz"
RECORD = ARTICLE / "data/articulated_ground_atlas.json"
OUTPUT = ARTICLE / "figures/fig_articulated_ground_atlas"
PALETTE = ("#4d4d4d", "#1f77b4", "#9467bd", "#d62728")


def _envelope(
    axis: plt.Axes,
    horizons_ms: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    scale: float = 1.0,
    linestyle: str = "-",
) -> None:
    samples = np.asarray(values).reshape(-1, values.shape[-1]) * scale
    median = np.median(samples, axis=0)
    low, high = np.min(samples, axis=0), np.max(samples, axis=0)
    axis.plot(
        horizons_ms,
        median,
        marker="o",
        color=color,
        linestyle=linestyle,
        label=label,
    )
    axis.fill_between(horizons_ms, low, high, color=color, alpha=0.12)


def _mechanism(axis: plt.Axes) -> None:
    axis.add_patch(Rectangle((-0.22, 0.02), 0.44, 0.12, color="0.25"))
    axis.plot([0.0, 0.0], [0.14, 0.72], color="0.2", linewidth=4)
    axis.plot([0.0, 0.38], [0.58, 0.82], color="0.2", linewidth=3)
    axis.plot([0.38, 0.75], [0.82, 0.58], color="#1f77b4", linewidth=3)
    axis.scatter([0.75], [0.58], s=55, color="#1f77b4", zorder=4)
    axis.annotate(
        "",
        xy=(-0.04, 0.48),
        xytext=(-0.38, 0.48),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#d62728"},
    )
    axis.annotate(
        "",
        xy=(0.0, 0.18),
        xytext=(0.0, -0.20),
        arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#2ca02c"},
    )
    moment = FancyArrowPatch(
        (-0.26, 0.18),
        (0.26, 0.18),
        connectionstyle="arc3,rad=-0.8",
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=2.0,
        color="#9467bd",
    )
    axis.add_patch(moment)
    axis.text(-0.39, 0.53, r"$F_x$", color="#d62728")
    axis.text(0.04, -0.10, r"$F_z$", color="#2ca02c")
    axis.text(0.0, 0.31, r"$M_y$", color="#9467bd", ha="center")
    axis.text(0.75, 0.66, "Club", ha="center", fontsize=8)
    axis.text(0.0, -0.25, "Passive Translation and Free Moment", ha="center")
    axis.set_xlim(-0.5, 0.9)
    axis.set_ylim(-0.32, 0.96)
    axis.set_aspect("equal")
    axis.axis("off")


def _pathway_speed(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    horizons = arrays["horizons_s"] * 1000.0
    values = arrays["primary_final_speed"][:, :, :, -1, :, :]
    fixed = values[:, 0:1]
    for slot, (label, color) in enumerate(
        zip(arrays["ground_activation_names"], PALETTE, strict=True)
    ):
        _envelope(
            axis,
            horizons,
            values[:, slot] - fixed[:, 0],
            label=str(label).replace("_", " ").title(),
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Nested Horizon (ms)")
    axis.set_ylabel("Speed Difference From Fixed Base (m/s)")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=7, ncol=2)


def _ground_response(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    horizons = arrays["horizons_s"] * 1000.0
    force = arrays["primary_peak_ground_force"][:, 3, :, -1, :, :]
    moment = arrays["primary_peak_intrinsic_moment"][:, 3, :, -1, :, :]
    _envelope(axis, horizons, force, label="Resultant Force", color="#2ca02c")
    twin = axis.twinx()
    _envelope(
        twin,
        horizons,
        moment,
        label="Intrinsic Free Moment",
        color="#9467bd",
        linestyle="--",
    )
    axis.set_xlabel("Nested Horizon (ms)")
    axis.set_ylabel("Peak Ground Force (N)")
    twin.set_ylabel("Peak Intrinsic Moment (N m)")
    axis.grid(alpha=0.22)
    lines = axis.lines + twin.lines
    axis.legend(lines, [line.get_label() for line in lines], fontsize=7)


def _controls(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    horizons = arrays["horizons_s"] * 1000.0
    coupled = arrays["primary_final_speed"][:, 3, :, -1, :, :]
    controls = arrays["control_final_speed"][:, :, :, -1, :, :]
    for slot, (label, color) in enumerate(
        zip(arrays["control_names"], ("#ff7f0e", "#17becf"), strict=True)
    ):
        _envelope(
            axis,
            horizons,
            coupled - controls[:, slot],
            label="Coupled Minus " + str(label).replace("_", " ").title(),
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Nested Horizon (ms)")
    axis.set_ylabel("Final-Speed Difference (m/s)")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=7)


def _matching(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    x = arrays["load_match_relative_error"].ravel()
    y = arrays["work_match_relative_error"].ravel()
    delta = arrays["matched_speed_difference"].ravel()
    limit = float(np.max(np.abs(delta))) or 1.0e-6
    scatter = axis.scatter(
        x,
        y,
        c=delta,
        cmap="coolwarm",
        norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        s=14,
        alpha=0.72,
    )
    axis.axvline(0.05, color="black", linestyle="--", linewidth=0.8)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Peak-Grip-Load Relative Difference")
    axis.set_ylabel("Dissipated-Work Relative Difference")
    axis.grid(alpha=0.20)
    colorbar = axis.figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("Coupled Minus Fixed Speed (m/s)")


def _numerics(
    axis: plt.Axes, arrays: dict[str, np.ndarray], record: dict[str, object]
) -> None:
    steps = arrays["time_steps_s"] * 1000.0
    axis.semilogy(
        steps,
        arrays["time_refinement"],
        marker="o",
        label="Worst Energy Residual",
    )
    axis.semilogy(
        steps[-1],
        np.max(arrays["primary_trajectory_parity"]),
        marker="s",
        label="Native Trajectory Parity",
    )
    axis.semilogy(
        steps[-1],
        np.max(arrays["primary_ground_force_parity"]),
        marker="D",
        label="Native Ground-Force Parity",
    )
    axis.invert_xaxis()
    axis.set_xlabel("Time Step (ms; Finer to the Right)")
    axis.set_ylabel("Registered Relative Error")
    axis.grid(alpha=0.22)
    results = record["results"]
    axis.text(
        0.03,
        0.04,
        f"Primary: {record['design']['primary_trajectory_count']} Traces\n"
        f"Controls: {record['design']['control_trajectory_count']} Traces\n"
        f"Matched Cells: {results['matched_load_work_cell_count']}",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    axis.legend(fontsize=7, loc="upper right")


def main() -> None:
    with np.load(DATA) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.1), constrained_layout=True)
    _mechanism(axes[0, 0])
    axes[0, 0].set_title("A. Finite-Base Ground Pathways")
    _ground_response(axes[0, 1], arrays)
    axes[0, 1].set_title("B. Coupled Ground Wrench")
    _pathway_speed(axes[0, 2], arrays)
    axes[0, 2].set_title("C. Pathway Killswitches")
    _controls(axes[1, 0], arrays)
    axes[1, 0].set_title("D. Independent Falsification Controls")
    _matching(axes[1, 1], arrays)
    axes[1, 1].set_title("E. Registered Load–Work Screen")
    _numerics(axes[1, 2], arrays, record)
    axes[1, 2].set_title("F. Refinement and Native Parity")
    fig.suptitle(
        "Finite Ground Reaction and Intrinsic Free Moment in the Articulated Model\n"
        "Synthetic Mechanism Reference; No Force-Plate or Human Calibration",
        fontsize=13,
    )
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    svg_path = OUTPUT.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
