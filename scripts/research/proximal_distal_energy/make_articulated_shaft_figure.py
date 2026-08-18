"""Render the mechanism and registered articulated-shaft atlas figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_shaft_atlas.npz"
RECORD = ARTICLE / "data/articulated_shaft_atlas.json"
OUTPUT = ARTICLE / "figures/fig_articulated_shaft_atlas"


def _mechanism(axis: plt.Axes) -> None:
    arc = np.linspace(0.0, 1.0, 80)
    axis.plot(np.zeros_like(arc), -arc, color="0.45", linestyle="--", label="Rigid")
    axis.plot(0.16 * arc**2, -arc, color="#1f77b4", linewidth=2.2, label="Bent")
    axis.scatter([0.0], [0.0], s=45, color="black", zorder=4)
    axis.scatter([0.16], [-1.0], s=65, color="#1f77b4", zorder=4)
    axis.annotate(
        r"Tip Bending $\eta_b$",
        xy=(0.16, -1.0),
        xytext=(0.43, -0.84),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4"},
        fontsize=9,
    )
    axis.plot([-0.12, 0.12], [-0.05, -0.05], color="0.25", linewidth=2.0)
    for x, start_y, end_y, label in (
        (-0.10, -0.13, 0.03, r"$F_L$"),
        (0.10, 0.03, -0.13, r"$F_R$"),
    ):
        axis.annotate(
            "",
            xy=(x, end_y),
            xytext=(x, start_y),
            arrowprops={"arrowstyle": "-|>", "color": "#d62728", "lw": 1.8},
        )
        axis.text(x, 0.07, label, ha="center", color="#d62728", fontsize=9)
    torque = FancyArrowPatch(
        (-0.13, -0.24),
        (0.13, -0.24),
        connectionstyle="arc3,rad=-0.85",
        arrowstyle="-|>",
        mutation_scale=12,
        color="#9467bd",
        linewidth=1.8,
    )
    axis.add_patch(torque)
    axis.text(0.0, -0.39, r"Twist $\phi$", ha="center", color="#9467bd")
    axis.text(0.0, 0.11, "Distributed Grip Root", ha="center", fontsize=9)
    axis.set_xlim(-0.28, 0.62)
    axis.set_ylim(-1.10, 0.18)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.legend(loc="lower left", fontsize=7, frameon=False)


def _envelope(
    axis: plt.Axes,
    horizons_ms: np.ndarray,
    values: np.ndarray,
    slots: tuple[int, ...],
    labels: tuple[str, ...],
    scale: float,
    linestyle: str,
    palette: tuple[str, ...],
) -> None:
    for slot, label, color in zip(slots, labels, palette, strict=True):
        samples = values[:, slot].reshape(-1, values.shape[-1]) * scale
        median = np.median(samples, axis=0)
        low, high = np.min(samples, axis=0), np.max(samples, axis=0)
        line = axis.plot(
            horizons_ms,
            median,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label,
        )[0]
        axis.fill_between(horizons_ms, low, high, color=line.get_color(), alpha=0.13)


def _responses(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    horizons = arrays["horizons_s"] * 1000.0
    bending = np.mean(arrays["maximum_tip_bending_m"][:, :, :, -1], axis=3)
    twist = np.mean(arrays["maximum_twist_angle_rad"][:, :, :, -1], axis=3)
    _envelope(
        axis,
        horizons,
        bending,
        (1, 3),
        ("Bending Only", "Coupled"),
        1000.0,
        "-",
        ("#1f77b4", "#ff7f0e"),
    )
    twin = axis.twinx()
    _envelope(
        twin,
        horizons,
        twist,
        (2, 3),
        ("Torsion Only", "Coupled Twist"),
        1000.0,
        "--",
        ("#9467bd", "#ff7f0e"),
    )
    axis.set_xlabel("Nested Horizon (ms)")
    axis.set_ylabel("Maximum Tip Bending (mm)")
    twin.set_ylabel("Maximum Twist (mrad)")
    axis.grid(alpha=0.25)
    lines = axis.lines + twin.lines
    axis.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="best")


def _matching(axis: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    x = arrays["load_match_relative_error"].ravel()
    y = arrays["work_match_relative_error"].ravel()
    delta = arrays["matched_final_speed_difference_m_s"].ravel()
    limit = float(np.max(np.abs(delta))) or 1.0e-6
    scatter = axis.scatter(
        x,
        y,
        c=delta,
        cmap="coolwarm",
        norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        s=16,
        alpha=0.75,
    )
    axis.axvline(0.05, color="black", linestyle="--", linewidth=0.9)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Peak-Load Relative Difference")
    axis.set_ylabel("Dissipated-Work Relative Difference")
    axis.grid(alpha=0.20)
    colorbar = axis.figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("Coupled Minus Rigid Final Speed (m/s)")


def _numerics(
    axis: plt.Axes, arrays: dict[str, np.ndarray], record: dict[str, object]
) -> None:
    steps = arrays["time_steps_s"] * 1000.0
    residual = arrays["time_refinement_worst_normalized_residual"]
    axis.semilogy(steps, residual, marker="o", label="Energy Residual")
    axis.semilogy(
        steps[-1],
        np.max(arrays["trajectory_relative_error"]),
        marker="s",
        label="Native Trajectory Parity",
    )
    axis.semilogy(
        steps[-1],
        np.max(arrays["force_relative_error"]),
        marker="D",
        label="Native Force Parity",
    )
    axis.invert_xaxis()
    axis.set_xlabel("Time Step (ms; Finer to the Right)")
    axis.set_ylabel("Registered Relative Error")
    axis.grid(alpha=0.25)
    structural = record["structural_authority"]
    axis.text(
        0.03,
        0.04,
        "FE Bending: "
        f"{structural['bending_frequency_hz']:.3f} Hz\n"
        "Declared Torsion: "
        f"{structural['torsion_frequency_hz']:.3f} Hz\n"
        "One-Mode Reference: 1 vs 6 Modes",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    axis.legend(fontsize=7, loc="upper right")


def main() -> None:
    with np.load(DATA) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.6), constrained_layout=True)
    _mechanism(axes[0, 0])
    axes[0, 0].set_title("A. Passive Shaft Coordinates and Grip Wrench")
    _responses(axes[0, 1], arrays)
    axes[0, 1].set_title("B. Bending and Twist Across Nested Horizons")
    _matching(axes[1, 0], arrays)
    axes[1, 0].set_title("C. Registered Load–Work Matching Screen")
    _numerics(axes[1, 1], arrays, record)
    axes[1, 1].set_title("D. Refinement and Native-Engine Parity")
    fig.suptitle(
        "Passive Articulated Shaft Bending and Torsion Under Distributed Grip Loading\n"
        "Synthetic Structural Reference; No Equipment or Human Calibration",
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
