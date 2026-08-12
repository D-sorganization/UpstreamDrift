"""Render publication figures for the spatial common-state experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from scripts.research.proximal_distal_energy.spatial_full_body import (
    build_spatial_model,
    evaluate_hand_wrenches,
    forward_kinematics,
    prescribed_state,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE_DIR = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = ARTICLE_DIR / "data"
FIGURE_DIR = ARTICLE_DIR / "figures"

COLORS = {
    "navy": "#17324D",
    "blue": "#2C7FB8",
    "orange": "#D95F0E",
    "green": "#238B45",
    "gray": "#657786",
    "red": "#B2182B",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    svg = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(pdf)
    fig.savefig(svg)
    text = svg.read_text(encoding="utf-8")
    svg.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return pdf, svg


def _plot_skeleton(ax, time_s: float, view_label: str) -> None:  # type: ignore[no-untyped-def]
    model = build_spatial_model()
    q, _, _ = prescribed_state(model, time_s)
    kin = forward_kinematics(model, q)
    for index, joint in enumerate(model.joints[:14]):
        parent = joint.parent
        if parent >= 0:
            points = np.vstack(
                [kin.joint_position_m[parent], kin.joint_position_m[index]]
            )
            ax.plot(
                points[:, 0], points[:, 1], points[:, 2], color=COLORS["navy"], lw=2
            )
    pelvis = kin.joint_position_m[1]
    for foot_y in (-0.16, 0.16):
        ax.plot(
            [pelvis[0], 0.0],
            [pelvis[1], foot_y],
            [pelvis[2], 0.02],
            color=COLORS["gray"],
            lw=2.2,
        )
    club_origin = kin.joint_position_m[model.club_frame_joint]
    club_rotation = kin.joint_rotation[model.club_frame_joint]
    clubhead = club_origin + club_rotation @ np.array([0.0, 0.0, -1.08])
    ax.plot(
        [club_origin[0], clubhead[0]],
        [club_origin[1], clubhead[1]],
        [club_origin[2], clubhead[2]],
        color=COLORS["orange"],
        lw=3,
    )
    sample = evaluate_hand_wrenches(model, time_s, coincident_hands=False)
    scale = 0.008
    for position, force, color, label in (
        (sample.lead_position_m, sample.lead_force_n, COLORS["blue"], "Lead"),
        (sample.trail_position_m, sample.trail_force_n, COLORS["red"], "Trail"),
    ):
        ax.quiver(
            *position,
            *(scale * force),
            color=color,
            linewidth=1.8,
            arrow_length_ratio=0.18,
        )
        ax.text(*(position + scale * force), label, color=color, fontsize=8)
    ax.scatter(*sample.reference_position_m, color="black", s=16, zorder=4)
    ax.set_title(f"{view_label}: $t={time_s:.3f}$ s")
    ax.set_xlabel("Target Axis, x (m)")
    ax.set_ylabel("Lateral Axis, y (m)")
    ax.set_zlabel("Vertical Axis, z (m)")
    ax.set_xlim(-0.15, 0.9)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(0.0, 1.7)
    ax.set_box_aspect((1.05, 1.1, 1.7))
    ax.view_init(elev=22, azim=-62 if view_label == "Oblique" else 28)


def make_geometry_figure() -> tuple[Path, Path]:
    """Show genuine nonplanar body motion and the named hand-force vectors."""

    _style()
    fig = plt.figure(figsize=(10.2, 5.5))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.20, top=0.84, wspace=0.10)
    left = fig.add_subplot(121, projection="3d")
    right = fig.add_subplot(122, projection="3d")
    _plot_skeleton(left, 0.215, "Oblique")
    _plot_skeleton(right, 0.215, "Cross-Target")
    fig.suptitle(
        "Spatial Full-Body Common State and Two-Hand Force Geometry", fontweight="bold"
    )
    fig.text(
        0.5,
        0.035,
        "Arrows are prescribed action–reaction contact loads; they are not solved muscle forces.",
        ha="center",
        color=COLORS["gray"],
        fontsize=8,
    )
    return _save(fig, "fig_spatial_full_body_force_geometry")


def make_cross_formulation_figure() -> tuple[Path, Path]:
    """Compare selected generalized actions and the residual envelope."""

    _style()
    with np.load(DATA_DIR / "spatial_full_body_study.npz") as data:
        time = data["time_s"]
        lagrange = data["inverse_dynamics_lagrange"]
        mujoco = data["inverse_dynamics_mujoco"]
    labels = ((2, "Torso Pitch"), (5, "Lead Shoulder Y"), (18, "Club Pitch"))
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), constrained_layout=True)
    for ax, (column, label) in zip(axes.flat[:3], labels, strict=True):
        ax.plot(
            time,
            lagrange[:, column],
            color=COLORS["navy"],
            lw=2,
            label="Lagrange–Christoffel",
        )
        ax.plot(
            time,
            mujoco[:, column],
            color=COLORS["orange"],
            lw=1.2,
            ls="--",
            label="MuJoCo",
        )
        ax.axvline(0.17, color=COLORS["gray"], lw=0.8, ls=":")
        ax.set_title(label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Generalized Force")
        ax.grid(alpha=0.2)
    error = np.abs(mujoco - lagrange)
    axes[1, 1].semilogy(time, np.max(error, axis=1), color=COLORS["red"], lw=1.8)
    axes[1, 1].axhline(0.75, color=COLORS["gray"], ls="--", label="Predeclared Bound")
    axes[1, 1].set_title("Maximum Cross-Engine Residual")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Absolute Generalized-Force Error")
    axes[1, 1].grid(alpha=0.2)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle(
        "Independent Inverse-Dynamics Formulations Agree at the Same Spatial State",
        fontweight="bold",
    )
    return _save(fig, "fig_spatial_cross_formulation_inverse_dynamics")


def make_falsification_figure() -> tuple[Path, Path]:
    """Show geometry interventions and the fail-closed claim boundary."""

    _style()
    record = json.loads((DATA_DIR / "spatial_full_body_study.json").read_text())
    with np.load(DATA_DIR / "spatial_full_body_study.npz") as data:
        time = data["time_s"]
        baseline = data["force_generated_couple_nm"]
        reverse = data["reverse_geometry_couple_nm"]
        coincident = data["coincident_hands_couple_nm"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    axes[0].plot(time, baseline, color=COLORS["red"], lw=2, label="Registered Geometry")
    axes[0].plot(
        time, reverse, color=COLORS["green"], lw=1.8, label="Reversed Moment Arm"
    )
    axes[0].plot(
        time,
        coincident,
        color=COLORS["gray"],
        lw=1.4,
        ls="--",
        label="Coincident Hands",
    )
    axes[0].axvline(0.17, color="black", lw=0.8, ls=":")
    axes[0].axhline(0.0, color="black", lw=0.7)
    axes[0].set_title("Contact Geometry, Not Force Norm, Sets Couple Sign")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Force-Generated Club Couple (N m)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    rows = [
        ("Spatial Inverse Dynamics", "Supported", COLORS["green"]),
        ("Geometry Sign Response", "Supported", COLORS["green"]),
        ("Passive Contact Origin", "Inconclusive", COLORS["orange"]),
        ("Forward Closed Contact", "Untested", COLORS["gray"]),
        ("Human Performance", "Unsupported", COLORS["red"]),
    ]
    axes[1].axis("off")
    axes[1].set_title("Claim Boundary After the Executed Experiment")
    y_values = np.linspace(0.84, 0.16, len(rows))
    for y, (claim, status, color) in zip(y_values, rows, strict=True):
        axes[1].text(0.03, y, claim, transform=axes[1].transAxes, va="center")
        axes[1].text(
            0.97,
            y,
            status,
            transform=axes[1].transAxes,
            ha="right",
            va="center",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": color,
                "edgecolor": "none",
            },
        )
    axes[1].text(
        0.03,
        0.02,
        "Cross-formulation relative residual: "
        f"{record['cross_formulation']['maximum_relative_inverse_dynamics_error']:.2e}",
        transform=axes[1].transAxes,
        color=COLORS["gray"],
        fontsize=8,
    )
    fig.suptitle(
        "Spatial Interventions Expose Both Support and Remaining Falsifiers",
        fontweight="bold",
    )
    return _save(fig, "fig_spatial_full_body_falsification")


def main() -> None:
    for paths in (
        make_geometry_figure(),
        make_cross_formulation_figure(),
        make_falsification_figure(),
    ):
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
