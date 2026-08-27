"""Render publication figures for the shaft-contribution study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"
COLORS = {
    "flex": "#007C91",
    "rigid": "#172B4D",
    "control": "#7C3AED",
    "momentum": "#D97706",
    "gravity": "#2A9D8F",
    "joint_damping": "#64748B",
    "shaft_elastic": "#B23A48",
    "shaft_damping": "#E76F51",
}


def _style() -> None:
    """Use deterministic portable vector-output settings."""
    plt.rcParams.update(
        {
            "pdf.use14corefonts": True,
            "svg.hashsalt": "upstreamdrift-shaft-contribution-v2",
            "axes.unicode_minus": False,
        }
    )


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    with (DATA_DIR / "shaft_contribution_study.json").open(encoding="utf-8") as stream:
        record = json.load(stream)
    return record, dict(np.load(DATA_DIR / "shaft_contribution_traces.npz"))


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    svg_path = FIG_DIR / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def fig_model_schematic() -> None:
    """Draw the exact rigid reduction and the flexible coordinate."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, flex, title in zip(
        axes,
        (0.0, -0.22),
        ("Rigid Coordinate Reduction", "One-Mode Flexible-Shaft Surrogate"),
        strict=True,
    ):
        theta1 = -0.75
        theta2 = 0.72
        angles = (theta1, theta1 + theta2, theta1 + theta2 + flex)
        lengths = (0.75, 0.45, 0.55)
        points = [np.zeros(2)]
        for angle, length in zip(angles, lengths, strict=True):
            points.append(
                points[-1] + length * np.array([np.sin(angle), -np.cos(angle)])
            )
        points_array = np.asarray(points)
        ax.plot(points_array[:3, 0], points_array[:3, 1], color=COLORS["rigid"], lw=5)
        ax.plot(points_array[2:, 0], points_array[2:, 1], color=COLORS["flex"], lw=5)
        ax.scatter(
            points_array[:, 0], points_array[:, 1], color="black", s=24, zorder=4
        )
        if flex:
            ax.annotate(
                r"$\phi_2$",
                xy=points_array[2],
                xytext=points_array[2] + np.array([0.12, 0.12]),
                arrowprops={"arrowstyle": "->", "color": COLORS["shaft_elastic"]},
                color=COLORS["shaft_elastic"],
                fontsize=12,
            )
            ax.text(0.04, 0.08, r"$U_s=\frac{1}{2}k\phi_2^2$", transform=ax.transAxes)
        else:
            ax.text(0.04, 0.08, r"$\phi_2=\dot\phi_2=0$", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 0.8)
        ax.set_ylim(-1.8, 0.2)
        ax.axis("off")
    fig.suptitle("Matched Mass Distribution With and Without a Shaft-Flex Coordinate")
    _save(fig, "fig_shaft_model_schematic")


def fig_matched_comparison(arrays: dict[str, np.ndarray], record: dict) -> None:
    """Compare matched rigid and flexible deliveries."""
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    for name, label, color in (
        ("flexible_reference", "Flexible", COLORS["flex"]),
        ("rigid_matched", "Rigid", COLORS["rigid"]),
    ):
        time = arrays[f"{name}__time"]
        axes[0].plot(time, arrays[f"{name}__tip_speed"], label=label, color=color)
    flex_time = arrays["flexible_reference__time"]
    flex_state = arrays["flexible_reference__state"]
    stiffness = record["reference_parameters"]["shaft_stiffness_nm_rad"]
    damping = record["reference_parameters"]["shaft_damping_nms_rad"]
    elastic = -stiffness * flex_state[:, 2]
    damped = -damping * flex_state[:, 5]
    axes[1].plot(flex_time, np.rad2deg(flex_state[:, 2]), color=COLORS["flex"])
    axes[2].plot(
        flex_time, elastic, color=COLORS["shaft_elastic"], label="Elastic Moment"
    )
    axes[2].plot(
        flex_time,
        damped,
        color=COLORS["shaft_damping"],
        ls="--",
        label="Damping Moment",
    )
    axes[0].set_ylabel("Clubhead Speed [m/s]")
    axes[1].set_ylabel("Shaft Flex [deg]")
    axes[2].set_ylabel("Moment [N m]")
    axes[2].set_xlabel("Time [s]")
    axes[0].legend()
    axes[2].legend()
    for ax in axes:
        ax.axhline(0.0, color="black", lw=0.7)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Matched Rigid and Flexible Deliveries Differ Modestly in the Reference Case"
    )
    _save(fig, "fig_shaft_matched_comparison")


def fig_acceleration_contributions(arrays: dict[str, np.ndarray]) -> None:
    """Show term-level contributions to absolute distal angular acceleration."""
    name = "flexible_reference"
    time = arrays[f"{name}__time"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    terms = (
        "control",
        "momentum",
        "gravity",
        "joint_damping",
        "shaft_elastic",
        "shaft_damping",
    )
    for term in terms:
        contribution = np.sum(arrays[f"{name}__accel_{term}"], axis=1)
        axes[0].plot(
            time, contribution, label=term.replace("_", " ").title(), color=COLORS[term]
        )
    axes[1].plot(
        time,
        np.sum(arrays[f"{name}__accel_total"], axis=1),
        color="black",
        lw=1.8,
        label="Total",
    )
    axes[0].set_ylabel("Contribution [rad/s²]")
    axes[1].set_ylabel("Distal Angular Acceleration [rad/s²]")
    axes[1].set_xlabel("Time [s]")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].legend()
    for ax in axes:
        ax.axhline(0.0, color="black", lw=0.7)
        ax.grid(alpha=0.2)
    fig.suptitle("Momentum, Gravity, Damping, Control, and Shaft Terms Sum Exactly")
    _save(fig, "fig_shaft_acceleration_contributions")


def fig_energy_closure(arrays: dict[str, np.ndarray]) -> None:
    """Show stored energies, supplied work, and first-law closure."""
    name = "flexible_reference"
    time = arrays[f"{name}__time"]
    kinetic = arrays[f"{name}__energy_kinetic_energy"]
    potential = arrays[f"{name}__energy_potential_energy"]
    strain = arrays[f"{name}__energy_shaft_strain_energy"]
    total = arrays[f"{name}__energy_total_mechanical_energy"]
    work = arrays[f"{name}__energy_cumulative_nonconservative_work"]
    closure = total - total[0] - work
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    axes[0].plot(time, kinetic, label="Kinetic")
    axes[0].plot(time, potential, label="Gravitational Potential")
    axes[0].plot(time, strain, label="Shaft Strain")
    axes[1].plot(
        time, total - total[0], label="Mechanical-Energy Change", color=COLORS["flex"]
    )
    axes[1].plot(
        time,
        work,
        label="Integrated External and Dissipative Power",
        color=COLORS["control"],
        ls="--",
    )
    axes[2].plot(time, closure * 1000.0, color=COLORS["shaft_elastic"])
    axes[0].set_ylabel("Energy [J]")
    axes[1].set_ylabel("Energy or Work [J]")
    axes[2].set_ylabel("Closure Error [mJ]")
    axes[2].set_xlabel("Time [s]")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Shaft Strain Energy Is Small but Explicitly Closed")
    _save(fig, "fig_shaft_energy_closure")


def fig_physics_ablations(record: dict) -> None:
    """Compare impact, work, and stored energy across physics ablations."""
    names = (
        "flexible_reference",
        "rigid_matched",
        "gravity_disabled",
        "joint_damping_disabled",
        "shaft_damping_disabled",
    )
    labels = ("Flexible", "Rigid", "No Gravity", "No Joint Damping", "No Shaft Damping")
    rows = {row["name"]: row for row in record["variant_summaries"]}
    metrics = (
        ("impact_speed_m_s", "Impact Speed [m/s]"),
        ("impact_time_s", "Impact Time [s]"),
        ("peak_shaft_strain_energy_j", "Peak Shaft Strain Energy [J]"),
        ("control_work_j", "Control Work [J]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    for ax, (key, title) in zip(axes.flat, metrics, strict=True):
        values = [
            (
                rows[name]["impact"][key]
                if key in rows[name]["impact"]
                else rows[name]["energy"][key]
            )
            for name in names
        ]
        ax.bar(
            np.arange(len(names)),
            values,
            color=[
                COLORS["flex"],
                COLORS["rigid"],
                COLORS["gravity"],
                COLORS["joint_damping"],
                COLORS["shaft_damping"],
            ],
        )
        ax.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Ablations Change Delivery More Than Reference Shaft Flex")
    _save(fig, "fig_shaft_physics_ablations")


def _grid(record: dict, metric: str, damping: float) -> tuple[np.ndarray, list, list]:
    grid = record["robustness_grid"]
    stiffness = grid["stiffness_values_nm_rad"]
    cuts = grid["cut_times_s"]
    rows = grid["rows"]
    values = np.array(
        [
            [
                next(
                    row[metric]
                    for row in rows
                    if row["shaft_stiffness_nm_rad"] == k
                    and row["shaft_damping_nms_rad"] == damping
                    and row["torque_cut_time_s"] == cut
                )
                for k in stiffness
            ]
            for cut in cuts
        ]
    )
    return values, stiffness, cuts


def fig_robustness_maps(record: dict) -> None:
    """Map stiffness and cut-time effects at reference shaft damping."""
    metrics = (
        ("impact_speed_m_s", "Impact Speed [m/s]", "viridis"),
        ("peak_shaft_strain_energy_j", "Peak Shaft Strain Energy [J]", "magma"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, (metric, title, cmap) in zip(axes, metrics, strict=True):
        values, stiffness, cuts = _grid(record, metric, damping=0.6)
        image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(stiffness)), [f"{value:.0f}" for value in stiffness])
        ax.set_yticks(
            range(len(cuts)),
            ["No Cut" if value is None else f"{value:.2f}" for value in cuts],
        )
        ax.set_xlabel("Shaft Stiffness [N m/rad]")
        ax.set_ylabel("Torque Cut Time [s]")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.84)
    fig.suptitle("Stiffness and Cut-Time Slices at Reference Damping (0.6 N m s/rad)")
    _save(fig, "fig_shaft_robustness_maps")


def fig_impact_window_and_timestep(record: dict) -> None:
    """Show endpoint-window dependence and timestep convergence."""
    rows = [
        row
        for row in record["robustness_grid"]["rows"]
        if row["shaft_damping_nms_rad"] == 0.6 and row["torque_cut_time_s"] is None
    ]
    rows.sort(key=lambda row: row["shaft_stiffness_nm_rad"])
    windows = record["robustness_grid"]["impact_windows_s"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for window in windows:
        key = f"peak_speed_within_{window * 1000:.0f}_ms_m_s"
        axes[0].plot(
            [row["shaft_stiffness_nm_rad"] for row in rows],
            [row[key] for row in rows],
            marker="o",
            label=f"±{window * 1000:.0f} ms" if window else "At Crossing",
        )
    axes[0].set_xlabel("Shaft Stiffness [N m/rad]")
    axes[0].set_ylabel("Scored Speed [m/s]")
    axes[0].set_title("Endpoint Window Changes the Reported Speed")
    axes[0].legend(fontsize=8)
    timestep_rows = sorted(record["timestep_rows"], key=lambda row: row["dt_s"])
    fine_speed = timestep_rows[0]["impact"]["impact_speed_m_s"]
    fine_time = timestep_rows[0]["impact"]["impact_time_s"]
    steps = np.array([row["dt_s"] for row in timestep_rows]) * 1000.0
    axes[1].plot(
        steps,
        [
            1000.0 * (row["impact"]["impact_speed_m_s"] - fine_speed)
            for row in timestep_rows
        ],
        marker="o",
        label="Speed Difference [mm/s]",
    )
    axes[1].plot(
        steps,
        [1e6 * (row["impact"]["impact_time_s"] - fine_time) for row in timestep_rows],
        marker="s",
        label="Time Difference [microseconds]",
    )
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_xlabel("Integrator Step [ms]")
    axes[1].set_ylabel("Difference From 0.25 ms")
    axes[1].set_title("Reference Delivery Converges With Timestep")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Impact Definition and Numerical Resolution")
    _save(fig, "fig_shaft_window_timestep")


def fig_flexible_pose_overlay(arrays: dict[str, np.ndarray]) -> None:
    """Overlay rigid and flexible shaft centerlines through six phases."""
    time = arrays["flexible_reference__time"]
    selected = (0.10, 0.20, 0.30, 0.40, 0.427, 0.459)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0))
    for ax, target in zip(axes.flat, selected, strict=True):
        index = int(np.argmin(np.abs(time - target)))
        for name, label, color, style in (
            ("rigid_matched", "Rigid", COLORS["rigid"], "--"),
            ("flexible_reference", "Flexible", COLORS["flex"], "-"),
        ):
            points = np.vstack(
                (
                    np.zeros(2),
                    arrays[f"{name}__wrist1"][index],
                    arrays[f"{name}__wrist2"][index],
                    arrays[f"{name}__tip"][index],
                )
            )
            ax.plot(
                points[:, 0], points[:, 1], color=color, ls=style, lw=2.3, label=label
            )
        ax.set_title(f"t = {time[index]:.3f} s")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Flexible and Rigid Centerlines Separate Near Delivery")
    _save(fig, "fig_shaft_pose_overlay")


def main() -> None:
    """Render all shaft-contribution figures as PDF and SVG."""
    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    record, arrays = _load()
    fig_model_schematic()
    fig_matched_comparison(arrays, record)
    fig_acceleration_contributions(arrays)
    fig_energy_closure(arrays)
    fig_physics_ablations(record)
    fig_robustness_maps(record)
    fig_impact_window_and_timestep(record)
    fig_flexible_pose_overlay(arrays)


if __name__ == "__main__":
    main()
