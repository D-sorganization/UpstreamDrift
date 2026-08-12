"""Render publication figures for the higher-order mechanism ladder."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from scripts.research.proximal_distal_energy.mechanism_ladder import (
    embed_planar_sample,
    rotation_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
COLORS = {
    "double": "#172B4D",
    "three": "#007C91",
    "mobile": "#2A9D8F",
    "loop": "#D97706",
    "three_d": "#7C3AED",
    "pending": "#94A3B8",
    "couple": "#B23A48",
}


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads(
        (DATA_DIR / "mechanism_ladder_study.json").read_text(encoding="utf-8")
    )
    return record, dict(np.load(DATA_DIR / "mechanism_ladder_traces.npz"))


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    svg_path = FIG_DIR / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def fig_ladder_schematic() -> None:
    """Show added mechanisms and evidence boundaries at every tier."""
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    tiers = [
        ("Two-Link\nPlanar", "Single Interface", COLORS["double"]),
        ("Three-Link\nPlanar", "Second Interface", COLORS["three"]),
        ("Moving\nHub", "Base Translation", COLORS["mobile"]),
        ("Two-Hand\nLoop", "Constraint Rank", COLORS["loop"]),
        ("Rotated 3-D\nWrench", "Frame Audit", COLORS["three_d"]),
        ("Reduced\nFull-Body\nDynamics", "Common State", COLORS["mobile"]),
        ("Forward Spatial\nContact", "Not Executed", COLORS["pending"]),
    ]
    for index, (title, subtitle, color) in enumerate(tiers):
        ax.add_patch(
            plt.Rectangle(
                (index - 0.42, -0.38),
                0.84,
                0.76,
                facecolor=color,
                edgecolor="white",
                linewidth=2,
                alpha=0.95,
            )
        )
        ax.text(
            index, 0.08, title, ha="center", va="center", color="white", weight="bold"
        )
        ax.text(
            index, -0.25, subtitle, ha="center", va="center", color="white", fontsize=9
        )
        if index < len(tiers) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (index + 0.43, 0.0),
                    (index + 0.57, 0.0),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=1.5,
                    color="#334155",
                )
            )
    ax.text(
        (len(tiers) - 1) / 2,
        -0.72,
        "Observables Stay Fixed: Reference Point, Frame, Force, Couple, Velocity, Angular Velocity, and Power",
        ha="center",
        color="#334155",
        fontsize=10,
    )
    ax.set_xlim(-0.65, len(tiers) - 0.35)
    ax.set_ylim(-0.9, 0.62)
    ax.axis("off")
    ax.set_title(
        "A Mechanism Ladder Must Add Degrees of Freedom Without Redefining Transfer"
    )
    _save(fig, "fig_ladder_schematic")


def fig_three_link_observables(arrays: dict[str, np.ndarray], record: dict) -> None:
    """Plot the common interface force and power fields."""
    time = arrays["time"]
    force = arrays["three_link__force"]
    total_power = arrays["three_link__power"]
    impact = record["three_link_reference"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    axes[0].plot(time, np.linalg.norm(force, axis=1), color=COLORS["three"], lw=2.2)
    axes[0].plot(time, force[:, 0], color="#64748B", lw=1.4, label="$F_x$")
    axes[0].plot(time, force[:, 1], color="#D97706", lw=1.4, label="$F_y$")
    axes[0].set_ylabel("Interface Force [N]")
    axes[0].legend(ncol=2)
    axes[1].plot(
        time, total_power, color=COLORS["three"], lw=2.2, label="Total Wrench Power"
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    for ax in axes:
        ax.axvline(impact["delivery_time_s"], color=COLORS["couple"], ls="--", lw=1.2)
        ax.grid(alpha=0.25)
    axes[1].set_ylabel("Power [W]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    fig.suptitle("The Same Reference-Explicit Observables Extend to a Third Coordinate")
    _save(fig, "fig_ladder_three_link_observables")


def fig_mobile_hub(arrays: dict[str, np.ndarray]) -> None:
    """Show how prescribed hub motion changes force and power attribution."""
    time = arrays["time"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for amplitude, color in (
        (0, COLORS["double"]),
        (50, COLORS["mobile"]),
        (100, COLORS["couple"]),
    ):
        key = f"hub_{amplitude:03d}mm"
        force = arrays[f"{key}__force"]
        axes[0].plot(
            time,
            np.linalg.norm(force, axis=1),
            color=color,
            lw=2,
            label=f"{amplitude} mm",
        )
        axes[1].plot(
            time, arrays[f"{key}__power"], color=color, lw=2, label=f"{amplitude} mm"
        )
    axes[0].set_title("Distal Interface Force")
    axes[0].set_ylabel("Force Magnitude [N]")
    axes[1].set_title("Distal Interface Wrench Power")
    axes[1].set_ylabel("Power [W]")
    for ax in axes:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.25)
        ax.legend(title="Hub Amplitude")
    fig.suptitle(
        "Prescribed Hub Motion Changes Reaction Force and Power Without Changing the Relative Trace"
    )
    _save(fig, "fig_ladder_mobile_hub")


def fig_closed_loop(arrays: dict[str, np.ndarray]) -> None:
    """Draw the planar loop and its constraint conditioning history."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axes[0]
    anchors = np.array([[-0.35, 0.0], [0.35, 0.0]])
    hands = np.array([[-0.12, -0.68], [0.13, -0.62]])
    grip_mid = hands.mean(axis=0)
    for anchor, hand, color, label in zip(
        anchors,
        hands,
        (COLORS["three"], COLORS["loop"]),
        ("Lead Arm", "Trail Arm"),
        strict=True,
    ):
        ax.plot(
            [anchor[0], hand[0]], [anchor[1], hand[1]], color=color, lw=6, label=label
        )
        ax.add_patch(Circle(anchor, 0.025, color="#172B4D"))
        ax.add_patch(Circle(hand, 0.022, color="black"))
    ax.plot(
        hands[:, 0],
        hands[:, 1],
        color="#334155",
        lw=8,
        solid_capstyle="round",
        label="Grip",
    )
    ax.scatter(*grip_mid, color=COLORS["couple"], s=45, zorder=5)
    ax.text(
        grip_mid[0] + 0.03,
        grip_mid[1],
        r"Grip Pose $(x,y,\psi)$",
        va="center",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.82, 0.12)
    ax.axis("off")
    ax.set_title("Four Contact Constraints on Five Coordinates")
    ax.legend(loc="lower left", fontsize=8)

    phase = arrays["closed_loop__phase"]
    condition = arrays["closed_loop__condition_number"]
    axes[1].plot(phase, condition, color=COLORS["loop"], lw=2.3)
    axes[1].fill_between(phase, condition, alpha=0.15, color=COLORS["loop"])
    axes[1].set_xlabel("Declared Downswing Phase")
    axes[1].set_ylabel("Constraint-Jacobian Condition Number")
    axes[1].set_title("Rank 4, Nullspace Dimension 1")
    axes[1].grid(alpha=0.25)
    fig.suptitle(
        "Closed-Loop Geometry Restricts Motion but Does Not Determine Contact Force"
    )
    _save(fig, "fig_ladder_closed_loop")


def fig_rotated_wrenches(arrays: dict[str, np.ndarray], record: dict) -> None:
    """Render the same delivery wrench in six proper 3-D frames."""
    time = arrays["time"]
    index = int(
        np.argmin(np.abs(time - record["three_link_reference"]["delivery_time_s"]))
    )
    point = arrays["three_link__point"][index, :2]
    force = arrays["three_link__force"][index, :2]
    sample = embed_planar_sample(
        model_tier="three-link-planar",
        time_s=float(time[index]),
        reference_point_xy_m=point,
        force_xy_n=force,
        couple_z_nm=record["three_link_reference"]["interface_couple_at_delivery_nm"],
        linear_velocity_xy_m_s=np.zeros(2),
        angular_velocity_z_rad_s=0.0,
    )
    fig = plt.figure(figsize=(11.5, 7.0))
    for panel, angle in enumerate(np.linspace(0.0, 2.2, 6)):
        ax = fig.add_subplot(2, 3, panel + 1, projection="3d")
        transform = rotation_matrix(np.array([1.0, 0.6, 0.3]), float(angle))
        rotated = sample.rotate(transform, frame=f"frame-{panel}")
        origin = rotated.reference_point_m
        vector = rotated.force_n / 90.0
        ax.quiver(
            *origin,
            *vector,
            color=COLORS["three_d"],
            linewidth=2.4,
            arrow_length_ratio=0.18,
        )
        ax.scatter(*origin, color="black", s=18)
        ax.plot(
            [0.0, origin[0]],
            [0.0, origin[1]],
            [0.0, origin[2]],
            color=COLORS["three"],
            lw=3,
        )
        ax.set_title(f"Frame {panel + 1}: {np.degrees(angle):.0f}°")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))
    fig.suptitle("A Proper 3-D Frame Rotation Changes Components, Not Wrench Power")
    _save(fig, "fig_ladder_rotated_wrenches")


def fig_invariance_residuals(record: dict) -> None:
    """Show numerical closure of frame and reference transformations."""
    audit = record["frame_and_transport_audits"]
    labels = ["Rotation\nPower", "Transport\nPower", "Force\nNorm", "Couple\nNorm"]
    values = [
        audit["maximum_rotation_power_residual_w"],
        audit["maximum_transport_power_residual_w"],
        audit["maximum_rotation_force_norm_residual_n"],
        audit["maximum_rotation_couple_norm_residual_nm"],
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(
        labels,
        values,
        color=[COLORS["three_d"], COLORS["loop"], COLORS["three"], COLORS["couple"]],
    )
    ax.set_yscale("log")
    ax.set_ylim(min(values) * 0.45, max(values) * 4.0)
    ax.set_ylabel("Maximum Absolute Residual")
    ax.set_title(
        "Frame and Reference-Transport Contracts Close at Floating-Point Scale"
    )
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.35,
            f"{value:.1e}",
            ha="center",
            fontsize=9,
        )
    _save(fig, "fig_ladder_invariance_residuals")


def fig_discrepancy_matrix(record: dict) -> None:
    """Render the executed-versus-open evidence boundary as a matrix."""
    rows = record["model_discrepancy_table"]
    mechanisms = [
        "Fixed Interface",
        "Third Coordinate",
        "Moving Hub",
        "Closed Loop",
        "3-D Frame",
        "Spatial Inverse Dynamics",
        "Forward Contact",
        "Articulated Contact",
    ]
    matrix = np.full((len(rows), len(mechanisms)), np.nan)
    for index in range(len(rows)):
        matrix[index, : index + 1] = 1.0
    matrix[-1, -1] = 0.0
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    cmap = matplotlib.colors.ListedColormap(["#E2E8F0", COLORS["three"]])
    ax.imshow(
        np.nan_to_num(matrix, nan=0.0), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto"
    )
    ax.set_xticks(range(len(mechanisms)), mechanisms, rotation=25, ha="right")
    tier_labels = [
        row["tier"]
        .replace("two_hand", "two-hand")
        .replace("three_link", "three-link")
        .replace("full_body", "full-body")
        .replace("cross_engine", "cross-engine")
        .replace("_", " ")
        .title()
        .replace("3D", "3-D")
        for row in rows
    ]
    ax.set_yticks(range(len(rows)), tier_labels)
    for row_index, row in enumerate(rows):
        for col_index in range(len(mechanisms)):
            if col_index > row_index:
                label = "—"
            elif row["status"] == "not_executed" and col_index == len(mechanisms) - 1:
                label = "Open"
            else:
                label = "Audited"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="#172B4D",
            )
    ax.set_title(
        "Model Discrepancy Matrix: Executed Mechanisms and the Remaining Boundary"
    )
    _save(fig, "fig_ladder_discrepancy_matrix")


def main() -> None:
    """Render all mechanism-ladder figures as PDF and SVG."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    record, arrays = _load()
    fig_ladder_schematic()
    fig_three_link_observables(arrays, record)
    fig_mobile_hub(arrays)
    fig_closed_loop(arrays)
    fig_rotated_wrenches(arrays, record)
    fig_invariance_residuals(record)
    fig_discrepancy_matrix(record)


if __name__ == "__main__":
    main()
