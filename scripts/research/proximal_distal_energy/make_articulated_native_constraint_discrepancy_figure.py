"""Render the native-versus-projected bilateral-contact discrepancy figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.deterministic_vector_figure import (
    save_vector_figure,
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
DATA = ARTICLE / "data/articulated_native_constraint_discrepancy.npz"
OUTPUT = ARTICLE / "figures/fig_articulated_native_constraint_discrepancy.pdf"

NATIVE = "#3B5CCC"
PROJECTED = "#D55E00"
DISCREPANCY = "#6A3D9A"


def make_figure(output: Path = OUTPUT) -> Path:
    """Render force, closure, and trajectory-discrepancy observables."""

    with np.load(DATA) as source:
        time_ms = np.asarray(source["time_s"], dtype=float) * 1000.0
        native_force = np.linalg.norm(
            source["native_generalized_constraint_force_n"], axis=1
        )
        projected_force = np.linalg.norm(
            source["projected_generalized_contact_force_n"], axis=1
        )
        native_separation = (
            np.asarray(source["native_attachment_separation_m"], dtype=float) * 1000.0
        )
        projected_separation = (
            np.asarray(source["projected_attachment_separation_m"], dtype=float)
            * 1000.0
        )
        trajectory_discrepancy = (
            np.max(
                np.abs(
                    np.asarray(source["native_q"], dtype=float)
                    - np.asarray(source["projected_q"], dtype=float)
                ),
                axis=1,
            )
            * 1000.0
        )
        steps_ms = np.asarray(source["time_step_s"], dtype=float) * 1000.0
        final_discrepancy = (
            np.asarray(source["final_trajectory_absolute_discrepancy"], dtype=float)
            * 1000.0
        )

    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), constrained_layout=True)
    force_axis, closure_axis, discrepancy_axis, refinement_axis = axes.flat
    force_axis.plot(
        time_ms,
        native_force,
        color=NATIVE,
        linewidth=2.2,
        label="MuJoCo Native Connect",
    )
    force_axis.plot(
        time_ms,
        projected_force,
        color=PROJECTED,
        linewidth=2.2,
        linestyle="--",
        label="Projected Kelvin-Voigt",
    )
    force_axis.set_title("A. Constraint and Contact Coordinate-Vector Norms")
    force_axis.set_ylabel("Coordinate-Vector Norm (Mixed Units)")
    force_axis.legend(frameon=False)

    closure_axis.plot(time_ms, native_separation, color=NATIVE, linewidth=2.2)
    closure_axis.plot(
        time_ms,
        projected_separation,
        color=PROJECTED,
        linewidth=2.2,
        linestyle="--",
    )
    closure_axis.set_title("B. Maximum Hand-Grip Separation")
    closure_axis.set_ylabel("Separation (mm)")

    discrepancy_axis.plot(
        time_ms, trajectory_discrepancy, color=DISCREPANCY, linewidth=2.2
    )
    discrepancy_axis.set_title("C. Native-Projected State Discrepancy")
    discrepancy_axis.set_xlabel("Time (ms)")
    discrepancy_axis.set_ylabel("Maximum Coordinate Difference (mm or rad × 1000)")

    refinement_axis.plot(
        steps_ms,
        final_discrepancy,
        color=DISCREPANCY,
        marker="o",
        linewidth=2.2,
    )
    refinement_axis.invert_xaxis()
    refinement_axis.set_title("D. Final Discrepancy Under Step Refinement")
    refinement_axis.set_xlabel("Time Step (ms; Finer to the Right)")
    refinement_axis.set_ylabel("Final Maximum Coordinate Difference (mm or rad × 1000)")

    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Native Equality Dynamics and Projected Compliant Contact Are Distinct Formulations",
        fontsize=14,
        fontweight="bold",
    )
    save_vector_figure(
        figure,
        output,
        salt="articulated-native-constraint-discrepancy-v1",
        write_svg=False,
        atomic_pdf=True,
    )
    plt.close(figure)
    return output


def main() -> None:
    print(make_figure())


if __name__ == "__main__":
    main()
