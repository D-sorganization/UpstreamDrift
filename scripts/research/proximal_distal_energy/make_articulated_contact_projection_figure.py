"""Render the articulated contact-projection qualification figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.deterministic_vector_figure import (
    save_vector_figure,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_contact_projection.npz"
OUTPUT = ARTICLE / "figures/fig_articulated_contact_projection"


def _heatmap(
    axis: plt.Axes, values: np.ndarray, title: str, label: str, *, log: bool = False
) -> None:
    plotted = np.log10(np.maximum(values, np.finfo(float).tiny)) if log else values
    image = axis.imshow(plotted, aspect="auto", origin="lower", cmap="viridis")
    axis.set_title(title)
    axis.set_xlabel("Closed-State Phase Sample")
    axis.set_ylabel("Profile-Span Case")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(label)


def main() -> int:
    """Write stable PDF and SVG views of the registered arrays."""

    with np.load(DATA) as source:
        force = source["maximum_contact_force_n"]
        acceleration_error = source["acceleration_relative_error"]
        zero_preload = source["zero_preload_force_n"]
        dissipation = -source["contact_dissipation_power_w"]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), constrained_layout=True)
    figure.suptitle(
        "Articulated Bilateral Contact Projection Passes All 234 Closed States\n"
        "Same-state acceleration only; no forward trajectory is integrated",
        fontsize=14,
    )
    _heatmap(axes[0, 0], force, "Maximum Contact Force", "Force (N)")
    _heatmap(
        axes[0, 1],
        acceleration_error,
        "Native Initial-Acceleration Relative Error",
        "log10 Relative Error",
        log=True,
    )
    _heatmap(
        axes[1, 0],
        zero_preload,
        "Zero-Preload Closure Leakage",
        "log10 Force (N)",
        log=True,
    )
    _heatmap(
        axes[1, 1],
        dissipation,
        "Nonnegative Dissipation Magnitude",
        "Dissipated Power Magnitude (W)",
    )
    save_vector_figure(
        figure,
        OUTPUT,
        salt="articulated-contact-projection-v1",
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
