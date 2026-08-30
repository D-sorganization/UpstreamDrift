"""Render the articulated same-state attribution evidence for #9151."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.deterministic_vector_figure import (
    save_vector_figure,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_drift_contact_attribution.npz"
OUTPUT = ARTICLE / "figures/fig_articulated_drift_contact_attribution"
COLORS = {
    "configuration": "#355f8d",
    "velocity": "#8f4e85",
    "contact": "#d17b0f",
    "active": "#5e6b73",
}


def _envelope(
    axis: plt.Axes,
    time_s: np.ndarray,
    values: np.ndarray,
    names: list[str],
    *,
    scale: float = 1.0,
) -> None:
    for index, name in enumerate(names):
        series = values[..., index] * scale
        lower = np.min(series, axis=(0, 2))
        median = np.median(series, axis=(0, 2))
        upper = np.max(series, axis=(0, 2))
        color = COLORS[name]
        axis.fill_between(time_s * 1000.0, lower, upper, color=color, alpha=0.13)
        axis.plot(
            time_s * 1000.0, median, color=color, linewidth=2.0, label=name.title()
        )


def main() -> int:
    """Write stable PDF and SVG views of the registered arrays."""

    with np.load(DATA) as source:
        time_s = np.asarray(source["time_s"], dtype=float)
        names = source["contribution_names"].tolist()
        acceleration_share = np.asarray(source["mass_metric_acceleration_share"])
        power = np.asarray(source["generalized_power_contribution_w"])
        engine_error = np.max(source["engine_contribution_relative_error"], axis=-1)
        acceleration_cancellation = np.asarray(
            source["acceleration_cancellation_index"]
        )
        power_cancellation = np.asarray(source["power_cancellation_index"])

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    figure.suptitle(
        "Articulated Same-State Drift and Contact Attribution\n"
        "Synthetic pointwise identity across 234 states; no forward persistence or human inference",
        fontsize=14,
    )

    _envelope(axes[0, 0], time_s, acceleration_share, names, scale=100.0)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Mass-Metric Acceleration Projection")
    axes[0, 0].set_xlabel("Closed-State Phase Time (ms)")
    axes[0, 0].set_ylabel("Signed Share (%)")
    axes[0, 0].legend(frameon=False, ncol=2)

    _envelope(axes[0, 1], time_s, power, names)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_yscale("symlog", linthresh=0.25)
    axes[0, 1].set_title("Generalized Power Contributions")
    axes[0, 1].set_xlabel("Closed-State Phase Time (ms)")
    axes[0, 1].set_ylabel("Power (W, Symmetric Log Scale)")

    image = axes[1, 0].imshow(
        np.log10(np.maximum(engine_error, np.finfo(float).tiny)),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[1, 0].set_title("Worst Native Contribution Discrepancy")
    axes[1, 0].set_xlabel("Closed-State Phase Sample")
    axes[1, 0].set_ylabel("Profile-Span Case")
    colorbar = figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    colorbar.set_label("log10 Relative Error")

    for values, label, color in (
        (acceleration_cancellation, "Acceleration", "#355f8d"),
        (power_cancellation, "Power", "#d17b0f"),
    ):
        lower = np.min(values, axis=(0, 2))
        median = np.median(values, axis=(0, 2))
        upper = np.max(values, axis=(0, 2))
        axes[1, 1].fill_between(time_s * 1000.0, lower, upper, color=color, alpha=0.15)
        axes[1, 1].plot(
            time_s * 1000.0, median, color=color, linewidth=2.0, label=label
        )
    axes[1, 1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].set_title("Signed-Contribution Cancellation")
    axes[1, 1].set_xlabel("Closed-State Phase Time (ms)")
    axes[1, 1].set_ylabel("Cancellation Index (1 = No Cancellation)")
    axes[1, 1].legend(frameon=False)

    save_vector_figure(
        figure,
        OUTPUT,
        salt="articulated-drift-contact-attribution-v1",
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
