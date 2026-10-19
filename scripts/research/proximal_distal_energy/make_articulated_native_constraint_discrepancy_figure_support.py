"""Plotting helpers for the native-versus-projected constraint discrepancy figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

NATIVE = "#3B5CCC"
PROJECTED = "#D55E00"
DISCREPANCY = "#6A3D9A"


@dataclass(frozen=True, slots=True)
class _DiscrepancySeries:
    time_ms: np.ndarray
    native_force: np.ndarray
    projected_force: np.ndarray
    native_separation: np.ndarray
    projected_separation: np.ndarray
    trajectory_discrepancy: np.ndarray
    steps_ms: np.ndarray
    final_discrepancy: np.ndarray


def load_discrepancy_series(data_path: Path) -> _DiscrepancySeries:
    """Load figure-ready series from the committed discrepancy NPZ."""
    if not data_path.is_file():
        raise FileNotFoundError(f"missing discrepancy arrays: {data_path}")
    with np.load(data_path) as source:
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
    return _DiscrepancySeries(
        time_ms=time_ms,
        native_force=native_force,
        projected_force=projected_force,
        native_separation=native_separation,
        projected_separation=projected_separation,
        trajectory_discrepancy=trajectory_discrepancy,
        steps_ms=steps_ms,
        final_discrepancy=final_discrepancy,
    )


def render_discrepancy_panels(
    figure: Figure, axes: np.ndarray, series: _DiscrepancySeries
) -> None:
    """Draw the four publication panels on the provided axes grid."""
    force_axis, closure_axis, discrepancy_axis, refinement_axis = axes.flat
    force_axis.plot(
        series.time_ms,
        series.native_force,
        color=NATIVE,
        linewidth=2.2,
        label="MuJoCo Native Connect",
    )
    force_axis.plot(
        series.time_ms,
        series.projected_force,
        color=PROJECTED,
        linewidth=2.2,
        linestyle="--",
        label="Projected Kelvin-Voigt",
    )
    force_axis.set_title("A. Constraint and Contact Coordinate-Vector Norms")
    force_axis.set_ylabel("Coordinate-Vector Norm (Mixed Units)")
    force_axis.legend(frameon=False)

    closure_axis.plot(
        series.time_ms, series.native_separation, color=NATIVE, linewidth=2.2
    )
    closure_axis.plot(
        series.time_ms,
        series.projected_separation,
        color=PROJECTED,
        linewidth=2.2,
        linestyle="--",
    )
    closure_axis.set_title("B. Maximum Hand-Grip Separation")
    closure_axis.set_ylabel("Separation (mm)")

    discrepancy_axis.plot(
        series.time_ms, series.trajectory_discrepancy, color=DISCREPANCY, linewidth=2.2
    )
    discrepancy_axis.set_title("C. Native-Projected State Discrepancy")
    discrepancy_axis.set_xlabel("Time (ms)")
    discrepancy_axis.set_ylabel("Maximum Coordinate Difference (mm or rad × 1000)")

    refinement_axis.plot(
        series.steps_ms,
        series.final_discrepancy,
        color=DISCREPANCY,
        marker="o",
        linewidth=2.2,
    )
    refinement_axis.invert_xaxis()
    refinement_axis.set_title("D. Final Discrepancy Under Step Refinement")
    refinement_axis.set_xlabel("Time Step (ms; Finer to the Right)")
    refinement_axis.set_ylabel("Final Maximum Coordinate Difference (mm or rad × 1000)")

    for axis in axes.flat:
        _style_axis(axis)

    figure.suptitle(
        "Native Equality Dynamics and Projected Compliant Contact Are Distinct Formulations",
        fontsize=14,
        fontweight="bold",
    )


def _style_axis(axis: Axes) -> None:
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
