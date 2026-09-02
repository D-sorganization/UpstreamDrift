"""Render the subject-scaled articulated inertia cross-engine audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.deterministic_vector_figure import (
    save_vector_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer"
STEM = STUDY_DIR / "figures/fig_articulated_inertia_cross_engine"


def _error_panel(axis: plt.Axes, values: np.ndarray, title: str) -> None:
    image = axis.imshow(
        np.log10(np.maximum(values, np.finfo(float).tiny)),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axis.set(
        title=title, xlabel="Closed-State Phase Sample", ylabel="Profile–Span Case"
    )
    axis.figure.colorbar(image, ax=axis, label="log₁₀ Relative Error")


def main() -> int:
    """Write a vector-first four-panel numerical qualification figure."""

    with np.load(STUDY_DIR / "data/articulated_inertia_cross_engine.npz") as source:
        mass = np.asarray(source["mass_matrix_relative_error"])
        bias = np.asarray(source["bias_relative_error"])
        inverse = np.asarray(source["inverse_dynamics_relative_error"])
        eigenvalue = np.asarray(source["minimum_mass_matrix_eigenvalue"])
    record = json.loads(
        (STUDY_DIR / "data/articulated_inertia_cross_engine.json").read_text(
            encoding="utf-8"
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    _error_panel(axes[0, 0], mass, "Mass-Matrix Relative Error")
    _error_panel(axes[0, 1], bias, "Bias-Force Relative Error")
    _error_panel(axes[1, 0], inverse, "Inverse-Dynamics Relative Error")
    sample = np.arange(eigenvalue.shape[0] * eigenvalue.shape[1])
    for engine_index, engine in enumerate(("MuJoCo", "Pinocchio")):
        axes[1, 1].plot(
            sample,
            eigenvalue[:, :, engine_index].ravel(),
            label=engine,
            linewidth=1.2,
        )
    axes[1, 1].set(
        title="Positive-Definite Mass-Matrix Margin",
        xlabel="Flattened Closed-State Index",
        ylabel="Minimum Eigenvalue",
        yscale="log",
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    result = record["results"]
    fig.suptitle(
        "Native Articulated Dynamics Agree Across All 234 Closed States\n"
        f"Worst inverse-dynamics relative error: "
        f"{result['maximum_inverse_dynamics_relative_error']:.2e}",
        fontsize=14,
    )
    save_vector_figure(
        fig,
        STEM,
        salt="articulated-inertia-cross-engine-v1",
        title="Articulated Inertia Cross-Engine Audit",
        bbox_inches=None,
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
