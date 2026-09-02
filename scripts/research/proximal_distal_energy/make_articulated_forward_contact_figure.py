"""Render the bounded articulated forward-contact qualification figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.deterministic_vector_figure import (
    save_vector_figure,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_forward_contact.npz"
OUTPUT = ARTICLE / "figures/fig_articulated_forward_contact"


def _variant_step_map(
    axis: plt.Axes,
    values: np.ndarray,
    variants: np.ndarray,
    steps_ms: np.ndarray,
    title: str,
    label: str,
    *,
    log: bool = False,
) -> None:
    reduced = np.max(values, axis=0)
    if reduced.ndim == 3:
        reduced = np.max(reduced, axis=-1)
    plotted = np.log10(np.maximum(reduced, np.finfo(float).tiny)) if log else reduced
    image = axis.imshow(plotted, aspect="auto", origin="lower", cmap="viridis")
    axis.set_title(title)
    axis.set_xticks(range(len(steps_ms)), [f"{value:g}" for value in steps_ms])
    axis.set_yticks(range(len(variants)), variants)
    axis.set_xlabel("Time Step (ms)")
    axis.set_ylabel("Registered Variant")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(label)


def main() -> int:
    """Write stable PDF and SVG views of the registered arrays."""

    with np.load(DATA) as source:
        variants = source["variant_names"]
        steps_ms = 1000.0 * source["time_steps_s"]
        separation = source["maximum_attachment_separation_m"] * 1000.0
        energy = source["normalized_work_energy_residual"]
        trajectory = source["trajectory_relative_error"]
        speed = source["final_club_translation_speed_m_s"]
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    figure.suptitle(
        "Bounded Articulated Bilateral-Attachment Forward Dynamics\n"
        "Five-millisecond synthetic screen; no human or coaching inference",
        fontsize=14,
    )
    _variant_step_map(
        axes[0, 0],
        separation,
        variants,
        steps_ms,
        "Worst Attachment Separation",
        "Separation (mm)",
    )
    _variant_step_map(
        axes[0, 1],
        energy,
        variants,
        steps_ms,
        "Worst Work--Energy Residual",
        "log10 Normalized Residual",
        log=True,
    )
    _variant_step_map(
        axes[1, 0],
        trajectory,
        variants,
        steps_ms,
        "MuJoCo--Pinocchio Trajectory Difference",
        "log10 Relative Error",
        log=True,
    )
    speed_envelope = np.ptp(speed, axis=(0, 3))
    image = axes[1, 1].imshow(
        speed_envelope, aspect="auto", origin="lower", cmap="cividis"
    )
    axes[1, 1].set_title("Across-State Final Club-Speed Range")
    axes[1, 1].set_xticks(range(len(steps_ms)), [f"{value:g}" for value in steps_ms])
    axes[1, 1].set_yticks(range(len(variants)), variants)
    axes[1, 1].set_xlabel("Time Step (ms)")
    axes[1, 1].set_ylabel("Registered Variant")
    colorbar = figure.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
    colorbar.set_label("Speed Range (m/s)")
    save_vector_figure(
        figure,
        OUTPUT,
        salt="articulated-forward-contact-v1",
    )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
