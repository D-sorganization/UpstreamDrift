"""Publication figures for matched torque allocation and transmission preload."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA = ARTICLE / "data" / "torque_allocation_preload_study.npz"
FIGURES = ARTICLE / "figures"

BLUE = "#2457A6"
ORANGE = "#C56A1A"
GREEN = "#2B7A4B"
PURPLE = "#7251A3"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "pdf.use14corefonts": True,
            "ps.useafm": True,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def _save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf")
    figure.savefig(FIGURES / f"{name}.svg")
    plt.close(figure)


def allocation_surface(arrays) -> None:
    angles = np.rad2deg(arrays["club_angles_rad"])
    fractions = arrays["wrist_fractions"]
    force = arrays["hand_force_rms_n"]
    effort = arrays["joint_torque_norm_nm"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for axis, values, title, label in (
        (axes[0], force, "Hand-Force Requirement", "RMS hand force (N)"),
        (axes[1], effort, "Generalized-Torque Requirement", "Joint-torque norm (N m)"),
    ):
        image = axis.pcolormesh(
            fractions,
            angles,
            values,
            shading="auto",
            cmap="viridis",
        )
        optimum = fractions[np.argmin(values, axis=1)]
        axis.plot(
            optimum,
            angles,
            color="white",
            linewidth=2.2,
            label="minimum on this metric",
        )
        axis.set_title(title)
        axis.set_xlabel("Direct-wrist allocation fraction")
        axis.set_ylabel("Club angle (deg)")
        axis.set_xlim(0.0, 1.0)
        axis.legend(loc="lower right", framealpha=0.9)
        figure.colorbar(image, ax=axis, label=label)
    figure.suptitle("Same Club Moment, Different Internal Demands", fontweight="bold")
    _save(figure, "fig_torque_allocation_geometry_surface")


def moment_closure(arrays) -> None:
    angles = np.rad2deg(arrays["club_angles_rad"])
    indices = [0, len(angles) // 2, len(angles) - 1]
    fractions = arrays["wrist_fractions"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    for index in indices:
        axes[0].plot(
            fractions,
            arrays["direct_wrist_moment_nm"][index],
            label=f"club angle {angles[index]:.0f} deg",
        )
        axes[1].plot(
            fractions,
            arrays["grip_force_couple_nm"][index],
            label=f"club angle {angles[index]:.0f} deg",
        )
    axes[0].set_title("Direct Wrist Moment")
    axes[1].set_title("Moment From the Two-Hand Force Couple")
    for axis in axes:
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.set_xlabel("Direct-wrist allocation fraction")
        axis.set_ylabel("Control moment about club center (N m)")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(
        "An 8 N m Club Task Does Not Identify Its Actuator Source", fontweight="bold"
    )
    _save(figure, "fig_torque_allocation_moment_closure")


def role_reversal(arrays) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    records = (
        ("persistent_arm_drive_preloaded", "Persistent Directions, Preloaded"),
        ("wrist_to_arm_role_reversal_preloaded", "Role Reversal, Preloaded"),
        ("persistent_arm_drive_relaxed", "Persistent Directions, Relaxed Start"),
        ("wrist_to_arm_role_reversal_relaxed", "Role Reversal, Relaxed Start"),
    )
    for axis, (name, title) in zip(axes.flat, records, strict=True):
        time_ms = 1000.0 * arrays[f"{name}_time_s"]
        axis.plot(
            time_ms,
            arrays[f"{name}_desired_net_torque_nm"],
            color="0.2",
            linestyle="--",
            label="desired net",
        )
        axis.plot(
            time_ms,
            arrays[f"{name}_transmitted_net_torque_nm"],
            color=PURPLE,
            linewidth=2,
            label="transmitted net",
        )
        axis.plot(
            time_ms,
            arrays[f"{name}_transmitted_arm_torque_nm"],
            color=BLUE,
            alpha=0.8,
            label="arm channel",
        )
        axis.plot(
            time_ms,
            arrays[f"{name}_transmitted_wrist_torque_nm"],
            color=ORANGE,
            alpha=0.8,
            label="wrist channel",
        )
        axis.axhline(0.0, color="0.3", linewidth=0.7)
        axis.set_title(title)
        axis.set_xlabel("Time after transition (ms)")
        axis.set_ylabel("Torque (N m)")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle(
        "Dead-Zone Transmission Makes Sign Reversal Observable", fontweight="bold"
    )
    _save(figure, "fig_torque_role_reversal_transmission")


def continuous_preparation(arrays) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), constrained_layout=True)
    records = (
        ("continuous_persistent_arm_drive", "Persistent Channel Directions"),
        ("continuous_wrist_to_arm_role_reversal", "Complete Role Reversal"),
    )
    for axis, (name, title) in zip(axes, records, strict=True):
        time_ms = 1000.0 * arrays[f"{name}_time_s"]
        axis.plot(
            time_ms,
            arrays[f"{name}_desired_net_torque_nm"],
            color="0.2",
            linestyle="--",
            label="desired net",
        )
        axis.plot(
            time_ms,
            arrays[f"{name}_transmitted_net_torque_nm"],
            color=PURPLE,
            linewidth=2,
            label="transmitted net",
        )
        axis.plot(
            time_ms,
            arrays[f"{name}_transmitted_arm_torque_nm"],
            color=BLUE,
            alpha=0.85,
            label="arm channel",
        )
        axis.plot(
            time_ms,
            arrays[f"{name}_transmitted_wrist_torque_nm"],
            color=ORANGE,
            alpha=0.85,
            label="wrist channel",
        )
        axis.axvline(0.0, color="0.1", linewidth=1.1, label="transition")
        axis.axhline(0.0, color="0.45", linewidth=0.7)
        axis.set_title(title)
        axis.set_ylabel("Torque (N m)")
        axis.grid(alpha=0.22)
    axes[-1].set_xlabel("Time Relative to Command Transition (ms)")
    axes[0].legend(ncol=3, fontsize=8)
    figure.suptitle(
        "Preparation History Carries Internal State Through Reversal",
        fontweight="bold",
    )
    _save(figure, "fig_torque_continuous_preparation")


def main() -> None:
    _style()
    with np.load(DATA) as arrays:
        allocation_surface(arrays)
        moment_closure(arrays)
        role_reversal(arrays)
        continuous_preparation(arrays)


if __name__ == "__main__":
    main()
