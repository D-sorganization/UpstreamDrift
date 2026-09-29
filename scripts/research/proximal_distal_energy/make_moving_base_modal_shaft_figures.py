"""Render publication figures for the forward distributed-shaft study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from scripts.research.proximal_distal_energy.moving_base_modal_shaft import (
    ModalShaftCouplingParams,
    kinematics,
    modal_shaft_basis,
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
    "red": "#B2182B",
    "violet": "#6A51A3",
    "gray": "#657786",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.use14corefonts": True,
            "pdf.compression": 9,
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
            "savefig.bbox": "tight",
            "svg.hashsalt": "moving-base-modal-shaft-v1",
        }
    )


def _save(figure: Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    svg = FIGURE_DIR / f"{stem}.svg"
    figure.savefig(pdf, metadata={"CreationDate": None})
    figure.savefig(svg, metadata={"Creator": "Open Research"})
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    plt.close(figure)
    return pdf, svg


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads((DATA_DIR / "moving_base_modal_shaft_study.json").read_text())
    with np.load(DATA_DIR / "moving_base_modal_shaft_study.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return record, arrays


def _shaft_curve(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params = ModalShaftCouplingParams.publication_default(mode_count=3)
    basis = modal_shaft_basis(params)
    alpha = q[8]
    direction = np.array([np.sin(alpha), -np.cos(alpha)])
    normal = np.array([np.cos(alpha), np.sin(alpha)])
    deflection = basis.mode_shapes @ q[9:]
    points = (
        q[6:8] + basis.locations_m[:, None] * direction + deflection[:, None] * normal
    )
    root = q[6:8][None, :]
    return np.concatenate((root[:, 0], points[:, 0])), np.concatenate(
        (root[:, 1], points[:, 1])
    )


def geometry_figure(arrays: dict[str, np.ndarray]) -> Figure:
    params = ModalShaftCouplingParams.publication_default(mode_count=3)
    time = arrays["baseline_time_s"]
    states = arrays["baseline_q"]
    contacts = arrays["baseline_contact_force_on_club_n"]
    selected = (0.12, 0.22, 0.245)
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    for axis, target in zip(axes, selected, strict=True):
        index = int(np.argmin(np.abs(time - target)))
        q = states[index]
        points = kinematics(q, params)
        shaft_x, shaft_y = _shaft_curve(q)
        axis.plot(shaft_x, shaft_y, color=COLORS["navy"], lw=2.4, label="Modal Shaft")
        for side, color, force in (
            ("right", COLORS["blue"], contacts[index, 0]),
            ("left", COLORS["red"], contacts[index, 1]),
        ):
            shoulder = points[f"{side}_shoulder"]
            elbow = points[f"{side}_elbow"]
            hand = points[f"{side}_hand"]
            axis.plot(
                [shoulder[0], elbow[0], hand[0]],
                [shoulder[1], elbow[1], hand[1]],
                "o-",
                color=color,
                lw=1.5,
                ms=3,
            )
            grip = points[f"{side}_grip"]
            axis.arrow(
                grip[0],
                grip[1],
                0.003 * force[0],
                0.003 * force[1],
                color=color,
                width=0.002,
                head_width=0.025,
                length_includes_head=True,
            )
        axis.scatter(*points["clubhead"], s=26, color=COLORS["orange"], zorder=5)
        axis.set_title(f"t = {time[index]:.3f} s")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_xlabel("Target Axis (m)")
    axes[0].set_ylabel("Vertical Axis (m)")
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(
        "Distributed Shaft Modes, Moving Base, and Solved Two-Hand Forces Evolve Together",
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        "Arrows are achieved contact forces (0.003 m/N); shaft properties are synthetic reference values.",
        ha="center",
        color=COLORS["gray"],
        fontsize=8,
    )
    return figure


def mode_comparison_figure(arrays: dict[str, np.ndarray]) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    for row, prefix, title in (
        (0, "smooth", "Smooth Downswing-Like Drive"),
        (1, "pulse", "Resolved 2 ms Wrist-Moment Pulse"),
    ):
        for mode_count, color in (
            (1, COLORS["orange"]),
            (3, COLORS["green"]),
            (6, COLORS["navy"]),
        ):
            key = f"{prefix}_modes_{mode_count}"
            time_ms = 1e3 * arrays[f"{key}_time_s"]
            axes[row, 0].plot(
                time_ms,
                1e3 * arrays[f"{key}_tip_deflection_m"],
                color=color,
                lw=1.5,
                label=f"{mode_count} Mode{'s' if mode_count != 1 else ''}",
            )
            axes[row, 1].plot(
                time_ms,
                1e3
                * np.linalg.norm(
                    arrays[f"{key}_clubhead_position_m"]
                    - arrays[f"{prefix}_modes_6_clubhead_position_m"],
                    axis=1,
                ),
                color=color,
                lw=1.5,
            )
        axes[row, 0].set_title(f"{title}: Tip Deflection")
        axes[row, 1].set_title(f"{title}: Position Error vs Six Modes")
        axes[row, 0].set_ylabel("Deflection (mm)")
        axes[row, 1].set_ylabel("Difference (mm)")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].grid(alpha=0.2)
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    figure.suptitle(
        "Mode Truncation Is Excitation-Dependent and Converges by Three Modes Here",
        fontweight="bold",
    )
    return figure


def killswitch_figure(record: dict, arrays: dict[str, np.ndarray]) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    time_ms = 1e3 * (arrays["branch_time_s"] - arrays["branch_time_s"][0])
    axes[0, 0].plot(
        time_ms,
        arrays["branch_force_couple_nm"],
        color=COLORS["red"],
        lw=1.8,
        label="Registered Arms",
    )
    axes[0, 0].plot(
        time_ms,
        arrays["branch_reversed_arm_couple_nm"],
        color=COLORS["green"],
        lw=1.4,
        label="Reversed Arms",
    )
    axes[0, 0].plot(
        time_ms,
        arrays["branch_coincident_couple_nm"],
        color=COLORS["navy"],
        ls="--",
        label="Coincident",
    )
    axes[0, 0].axhline(0.0, color="black", lw=0.8)
    axes[0, 0].set_title("Same-State Zero-Command Geometry Controls")
    axes[0, 0].set_ylabel("Force Couple (N m)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(
        time_ms,
        1e3 * arrays["branch_tip_deflection_m"],
        color=COLORS["violet"],
        lw=1.7,
    )
    axes[0, 1].set_title("Shaft Deflection Remains Dynamic After the Cut")
    axes[0, 1].set_ylabel("Tip Deflection (mm)")
    steps = np.array([row["step_s"] for row in record["timestep_refinement"]])
    residuals = np.array(
        [
            row["closure"]["work_energy_residual_abs_j"]
            for row in record["timestep_refinement"]
        ]
    )
    axes[1, 0].loglog(1e3 * steps, residuals, "o-", color=COLORS["blue"])
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_title("Work--Energy Residual Decreases with Timestep")
    axes[1, 0].set_xlabel("Timestep (ms)")
    axes[1, 0].set_ylabel("Residual (J)")
    baseline_time = arrays["baseline_time_s"]
    axes[1, 1].plot(
        baseline_time,
        arrays["baseline_strain_energy_j"],
        color=COLORS["orange"],
        lw=1.5,
        label="Modal Strain Energy",
    )
    axes[1, 1].axvline(0.22, color=COLORS["gray"], ls=":", label="Branch State")
    axes[1, 1].set_title("Baseline Modal Storage Before the Intervention")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Negative Couple Persists After Command Removal and Survives Numerical Controls",
        fontweight="bold",
    )
    return figure


def main() -> None:
    _style()
    record, arrays = _load()
    for figure, stem in (
        (geometry_figure(arrays), "fig_modal_shaft_forward_geometry"),
        (mode_comparison_figure(arrays), "fig_modal_shaft_mode_convergence"),
        (killswitch_figure(record, arrays), "fig_modal_shaft_killswitch_controls"),
    ):
        for path in _save(figure, stem):
            print(path)


if __name__ == "__main__":
    main()
