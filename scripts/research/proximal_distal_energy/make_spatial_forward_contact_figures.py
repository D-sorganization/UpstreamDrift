"""Render publication figures for the two-engine spatial contact study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


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
    "violet": "#6A51A3",
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
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
            "pdf.use14corefonts": True,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    svg_path = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return pdf_path, svg_path


def _load() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    record = json.loads((DATA_DIR / "spatial_forward_contact_study.json").read_text())
    with np.load(DATA_DIR / "spatial_forward_contact_study.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return record, arrays


def _equalize_3d(ax) -> None:  # type: ignore[no-untyped-def]
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * float(np.max(limits[:, 1] - limits[:, 0]))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def make_geometry_montage() -> tuple[Path, Path]:
    """Show achieved 3-D geometry, contact forces, and club-axis evolution."""

    _style()
    _, data = _load()
    time = data["mujoco_baseline_time"]
    requested_times = (0.090, 0.180, 0.225)
    indices = [int(np.argmin(np.abs(time - value))) for value in requested_times]
    fig = plt.figure(figsize=(10.4, 4.2))
    axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    force_scale = 0.008
    for panel, (ax, sample_index) in enumerate(zip(axes, indices, strict=True)):
        center = data["mujoco_baseline_club_position"][sample_index]
        club_axis = data["mujoco_baseline_club_axis"][sample_index]
        hand_positions = data["mujoco_baseline_hand_positions"][sample_index]
        points = data["mujoco_baseline_contact_points"][sample_index]
        forces = data["mujoco_baseline_contact_forces"][sample_index]
        shaft = np.vstack([center - 0.42 * club_axis, center + 0.42 * club_axis])
        ax.plot(*shaft.T, color=COLORS["navy"], lw=4, label="Club Axis")
        ax.plot(*points.T, color=COLORS["gray"], lw=1.0, ls=":")
        for hand_index, color, label in (
            (0, COLORS["blue"], "Lead Interface"),
            (1, COLORS["red"], "Trail Interface"),
        ):
            hand = hand_positions[hand_index]
            point = points[hand_index]
            force = forces[hand_index]
            ax.plot(*np.vstack([hand, point]).T, color=color, lw=1.6)
            ax.scatter(*hand, color=color, s=24, label=label if panel == 0 else None)
            ax.quiver(
                *point,
                *(force_scale * force),
                color=color,
                linewidth=1.7,
                arrow_length_ratio=0.18,
            )
        ax.scatter(*center, color="black", s=18)
        ax.set_title(f"$t={time[sample_index]:.3f}$ s")
        ax.set_xlabel("Target Axis (m)")
        ax.set_ylabel("Lateral Axis (m)")
        if panel == 0:
            ax.set_zlabel("Vertical Axis (m)")
        ax.view_init(elev=24, azim=-62)
        _equalize_3d(ax)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        "Achieved Spatial Contact Geometry and Engine-Solved Force Vectors",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Arrows are compliant interface forces from achieved relative state (0.008 m/N); no force or torque is applied directly to the club.",
        ha="center",
        color=COLORS["gray"],
        fontsize=8,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.20, top=0.82, wspace=0.08)
    return _save(fig, "fig_spatial_forward_contact_geometry")


def make_cross_engine_figure() -> tuple[Path, Path]:
    """Show forward trajectory, wrench, orientation, and numerical parity."""

    _style()
    record, data = _load()
    time = data["mujoco_baseline_time"]
    mujoco_position = data["mujoco_baseline_club_position"]
    pinocchio_position = data["pinocchio_baseline_club_position"]
    mujoco_wrench = data["mujoco_baseline_contact_wrench"]
    pinocchio_wrench = data["pinocchio_baseline_contact_wrench"]
    metrics = record["numerical_gates"]["baseline"]["observed_metrics"]  # type: ignore[index]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.1), constrained_layout=True)
    labels = ("x", "y", "z")
    for column, label, color in zip(
        range(3),
        labels,
        (COLORS["navy"], COLORS["blue"], COLORS["orange"]),
        strict=True,
    ):
        axes[0, 0].plot(
            time, mujoco_position[:, column], color=color, lw=2, label=label
        )
        axes[0, 0].plot(time, pinocchio_position[:, column], color=color, lw=1, ls="--")
    axes[0, 0].set_title("Club Translation in Two Native Engines")
    axes[0, 0].set_ylabel("Position (m)")
    axes[0, 0].legend(frameon=False, ncol=3, title="MuJoCo Solid; Pinocchio Dashed")

    axes[0, 1].plot(
        time,
        data["mujoco_baseline_swing_normal_couple"],
        color=COLORS["red"],
        lw=2,
        label="MuJoCo",
    )
    axes[0, 1].plot(
        time,
        data["pinocchio_baseline_swing_normal_couple"],
        color=COLORS["navy"],
        lw=1.2,
        ls="--",
        label="Pinocchio",
    )
    axes[0, 1].axhline(0.0, color="black", lw=0.7)
    axes[0, 1].set_title("Force-Generated Swing-Normal Couple")
    axes[0, 1].set_ylabel("Couple (N m)")
    axes[0, 1].legend(frameon=False)

    position_error = np.linalg.norm(mujoco_position - pinocchio_position, axis=1)
    wrench_error = np.linalg.norm(mujoco_wrench - pinocchio_wrench, axis=1)
    axes[1, 0].semilogy(
        time, np.maximum(position_error, 1e-12), color=COLORS["green"], lw=1.8
    )
    axes[1, 0].axhline(0.009, color=COLORS["gray"], ls="--", label="Declared Maximum")
    axes[1, 0].set_title("Forward Club-Position Divergence")
    axes[1, 0].set_ylabel("Euclidean Difference (m)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(time, wrench_error, color=COLORS["violet"], lw=1.8)
    axes[1, 1].set_title("Complete Contact-Wrench Difference")
    axes[1, 1].set_ylabel("Wrench-Norm Difference")
    axes[1, 1].text(
        0.03,
        0.94,
        "Position RMS: "
        f"{1e6 * metrics['club_position_rms_m']:.1f} µm\n"
        "Relative Wrench RMS: "
        f"{100 * metrics['contact_wrench_relative_rms']:.3f}%",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    for ax in axes.flat:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Independent MuJoCo and Pinocchio Forward Dynamics Satisfy Declared Gates",
        fontweight="bold",
    )
    return _save(fig, "fig_spatial_forward_cross_engine")


def make_killswitch_figure() -> tuple[Path, Path]:
    """Show the same-state branch, negative couple, and pathway observables."""

    _style()
    record, data = _load()
    time = data["mujoco_baseline_time"]
    kill_time = record["interventions"]["same_state_driver_killswitch_s"]  # type: ignore[index]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.1), constrained_layout=True)
    axes[0, 0].plot(
        time,
        data["mujoco_baseline_swing_normal_couple"],
        color=COLORS["gray"],
        lw=1.5,
        label="Driver Continues",
    )
    for engine, color in (("mujoco", COLORS["red"]), ("pinocchio", COLORS["navy"])):
        axes[0, 0].plot(
            time,
            data[f"{engine}_killswitch_swing_normal_couple"],
            color=color,
            lw=1.8,
            ls="--" if engine == "pinocchio" else "-",
            label=f"{engine.title()} Killswitch",
        )
    axes[0, 0].axhline(0.0, color="black", lw=0.7)
    axes[0, 0].set_title("Same-State Driver Killswitch Retains a Negative Couple")
    axes[0, 0].set_ylabel("Swing-Normal Couple (N m)")
    axes[0, 0].legend(frameon=False, fontsize=8)

    driver_norm = np.linalg.norm(data["mujoco_killswitch_driver_forces"], axis=2)
    axes[0, 1].plot(time, driver_norm[:, 0], color=COLORS["blue"], label="Lead Driver")
    axes[0, 1].plot(
        time, driver_norm[:, 1], color=COLORS["orange"], label="Trail Driver"
    )
    axes[0, 1].set_title("Grounded Driver Forces Are Exactly Removed")
    axes[0, 1].set_ylabel("Force Magnitude (N)")
    axes[0, 1].legend(frameon=False)

    long_axis_rate = np.sum(
        data["mujoco_killswitch_club_angular_velocity"]
        * data["mujoco_killswitch_club_axis"],
        axis=1,
    )
    axes[1, 0].plot(time, long_axis_rate, color=COLORS["violet"], lw=1.8)
    axes[1, 0].plot(
        time,
        np.rad2deg(data["mujoco_killswitch_swing_plane_tilt"]),
        color=COLORS["green"],
        lw=1.5,
        label="Swing-Plane Tilt (deg)",
    )
    axes[1, 0].set_title("Long-Axis Rotation and Swing-Plane Evolution")
    axes[1, 0].set_ylabel("Rate (rad/s) or Angle (deg)")
    axes[1, 0].legend(frameon=False)

    ground = data["mujoco_killswitch_ground_pathway_wrench"]
    axes[1, 1].plot(
        time,
        np.linalg.norm(ground[:, :3], axis=1),
        color=COLORS["navy"],
        label="Ground-Path Force",
    )
    axes[1, 1].plot(
        time,
        np.linalg.norm(ground[:, 3:], axis=1),
        color=COLORS["orange"],
        label="Ground-Path Moment",
    )
    axes[1, 1].set_title("Reduced Ground-Pathway Reaction Proxy")
    axes[1, 1].set_ylabel("Force (N) or Moment (N m)")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.axvline(float(kill_time), color="black", lw=0.8, ls=":")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Post-Killswitch Interaction Dynamics Persist Without Direct Club Actuation",
        fontweight="bold",
    )
    return _save(fig, "fig_spatial_forward_killswitch")


def make_energy_controls_figure() -> tuple[Path, Path]:
    """Show energy closure, geometry controls, and claim boundary."""

    _style()
    record, data = _load()
    time = data["mujoco_killswitch_time"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), constrained_layout=True)
    for engine, color in (("mujoco", COLORS["red"]), ("pinocchio", COLORS["navy"])):
        axes[0].plot(
            time,
            data[f"{engine}_killswitch_energy_balance_residual"],
            color=color,
            lw=1.5,
            label=engine.title(),
        )
    axes[0].set_title("Work–Energy Residual")
    axes[0].set_ylabel("Residual (J)")
    axes[0].legend(frameon=False)

    baseline = data["mujoco_baseline_swing_normal_couple"]
    reverse = data["mujoco_baseline_reversed_couple"]
    coincident = data["mujoco_baseline_coincident_couple"]
    axes[1].plot(time, baseline, color=COLORS["red"], lw=1.8, label="Registered")
    axes[1].plot(time, reverse, color=COLORS["green"], lw=1.5, label="Reversed Arms")
    axes[1].plot(time, coincident, color=COLORS["gray"], ls="--", label="Coincident")
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_title("Geometry Falsification Controls")
    axes[1].set_ylabel("Swing-Normal Couple (N m)")
    axes[1].legend(frameon=False, fontsize=8)

    rows = (
        ("Forward Contact", "Supported", COLORS["green"]),
        ("Two-Engine Transport", "Supported", COLORS["green"]),
        ("Anatomical Arms", "Untested", COLORS["gray"]),
        ("Muscle Mechanism", "Untested", COLORS["gray"]),
        ("Human Strategy", "Untested", COLORS["gray"]),
    )
    axes[2].axis("off")
    axes[2].set_title("Claim Boundary")
    for y, (claim, status, color) in zip(
        np.linspace(0.84, 0.18, len(rows)), rows, strict=True
    ):
        axes[2].text(0.03, y, claim, transform=axes[2].transAxes, va="center")
        axes[2].text(
            0.97,
            y,
            status,
            transform=axes[2].transAxes,
            ha="right",
            va="center",
            color="white",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "edgecolor": "none"},
        )
    duration = record["mechanism_tests"]["same_state_killswitch_negative_duration_s"]  # type: ignore[index]
    axes[2].text(
        0.03,
        0.02,
        f"Registered post-killswitch negative duration: {1000 * duration:.1f} ms",
        transform=axes[2].transAxes,
        fontsize=8,
        color=COLORS["gray"],
    )
    for ax in axes[:2]:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Conservation and Negative Controls Bound the Forward-Contact Claim",
        fontweight="bold",
    )
    return _save(fig, "fig_spatial_forward_energy_controls")


def main() -> None:
    for pair in (
        make_geometry_montage(),
        make_cross_engine_figure(),
        make_killswitch_figure(),
        make_energy_controls_figure(),
    ):
        for path in pair:
            print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "make_cross_engine_figure",
    "make_energy_controls_figure",
    "make_geometry_montage",
    "make_killswitch_figure",
]
