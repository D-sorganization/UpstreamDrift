"""Create publication figures for the advanced frame and biological bridge."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
FIGURE_DIR = ARTICLE / "figures"
DATA_DIR = ARTICLE / "data"

INK = "#17223B"
MUTED = "#5B6578"
GRID = "#D8DEE9"
PAPER = "#FCFCFD"
DRIFT = "#0072B2"
CONTROL = "#D55E00"
TRANSFER = "#009E73"
DISTAL = "#CC79A7"
WARNING = "#E69F00"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.0,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
    plt.close(fig)


def _arrow(
    ax: plt.Axes, start: np.ndarray, end: np.ndarray, color: str, label: str
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2.2,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    midpoint = 0.5 * (start + end)
    ax.text(
        midpoint[0], midpoint[1] + 0.08, label, color=color, ha="center", weight="bold"
    )


def make_frame_power_invariance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle("One Physical Interaction, Multiple Valid Descriptions", weight="bold")
    ax = axes[0]
    ax.set_title("Wrench and Twist at Two Reference Points")
    point_a = np.array([0.0, 0.0])
    point_b = np.array([1.35, 0.45])
    ax.plot(
        [point_a[0], point_b[0]], [point_a[1], point_b[1]], color=INK, lw=5, alpha=0.20
    )
    for point, label in ((point_a, "$A$"), (point_b, "$B$")):
        ax.add_patch(Circle(point, 0.07, color=INK, zorder=5))
        ax.text(point[0], point[1] - 0.18, label, ha="center", weight="bold")
    _arrow(ax, point_a, point_a + np.array([0.62, 0.72]), DRIFT, "$\\mathbf{F}$")
    _arrow(ax, point_a, point_a + np.array([0.75, -0.22]), TRANSFER, "$\\mathbf{v}_A$")
    _arrow(ax, point_b, point_b + np.array([0.34, 0.58]), DRIFT, "Same $\\mathbf{F}$")
    _arrow(ax, point_b, point_b + np.array([0.68, 0.00]), TRANSFER, "$\\mathbf{v}_B$")
    ax.annotate(
        "$\\mathbf{M}_B=\\mathbf{M}_A-(\\mathbf{r}_B-\\mathbf{r}_A)\\times\\mathbf{F}$",
        (0.67, 0.52),
        xytext=(0.38, 1.12),
        ha="center",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": GRID},
    )
    ax.text(
        0.68,
        -0.48,
        "$P=\\mathbf{F}\\cdot\\mathbf{v}+\\mathbf{M}\\cdot\\boldsymbol{\\omega}$",
        ha="center",
        fontsize=11,
    )
    ax.set_xlim(-0.3, 2.25)
    ax.set_ylim(-0.65, 1.48)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    ax.set_title("Executable Invariance Residuals")
    record = json.loads((DATA_DIR / "advanced_biological_bridge.json").read_text())
    frame = record["frame_invariance"]
    labels = ["Rotated Frame", "Shifted Point", "Jacobian Virtual Work"]
    values = [
        frame["maximum_rotation_power_residual_w"],
        frame["maximum_transport_power_residual_w"],
        frame["maximum_virtual_work_residual_w"],
    ]
    bars = ax.barh(labels, values, color=[DRIFT, TRANSFER, DISTAL], height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(1e-16, 1e-11)
    ax.set_xlabel("Maximum Absolute Power Residual (W)")
    ax.axvline(1e-11, color=WARNING, lw=1.5, ls="--", label="Declared Gate")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value * 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.1e}", va="center"
        )
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_frame_power_invariance")


def make_biological_redundancy(arrays: np.lib.npyio.NpzFile) -> None:
    coactivation = arrays["redundancy__coactivation"]
    positive = arrays["redundancy__positive_activation"]
    negative = arrays["redundancy__negative_activation"]
    torque = arrays["redundancy__net_torque_nm"]
    stiffness = arrays["redundancy__stiffness_proxy_nm_rad"]
    energy = arrays["redundancy__series_elastic_energy_j"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    fig.suptitle(
        "Muscle Redundancy: The Same Joint Moment Does Not Identify One Activation Pattern",
        weight="bold",
    )
    axes[0].plot(
        coactivation, positive, color=TRANSFER, lw=2.4, label="Positive-Moment Channel"
    )
    axes[0].plot(
        coactivation, negative, color=DISTAL, lw=2.4, label="Negative-Moment Channel"
    )
    axes[0].fill_between(coactivation, negative, positive, color=DRIFT, alpha=0.10)
    axes[0].set(
        title="Activation Family", xlabel="Antagonist Coactivation", ylabel="Activation"
    )
    axes[0].legend(fontsize=7.5)
    axes[1].plot(coactivation, torque, color=INK, lw=2.8)
    axes[1].axhline(10.0, color=WARNING, ls="--", lw=1.5)
    axes[1].set(
        title="Matched Mechanical Task",
        xlabel="Antagonist Coactivation",
        ylabel="Net Joint Moment (N m)",
    )
    axes[2].plot(coactivation, stiffness, color=DRIFT, lw=2.4, label="Stiffness Proxy")
    twin = axes[2].twinx()
    twin.plot(
        coactivation, energy, color=CONTROL, lw=2.4, label="Series-Elastic Energy"
    )
    axes[2].set(
        title="Internal Consequences",
        xlabel="Antagonist Coactivation",
        ylabel="Stiffness Proxy (N m/rad)",
    )
    twin.set_ylabel("Series-Elastic Energy (J)", color=CONTROL)
    lines = axes[2].lines + twin.lines
    axes[2].legend(
        lines, [line.get_label() for line in lines], fontsize=7.5, loc="upper left"
    )
    fig.tight_layout()
    _save(fig, "fig_biological_redundancy")


def make_biological_role_reversal(arrays: np.lib.npyio.NpzFile) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4), sharex=True)
    fig.suptitle(
        "Continuous Muscle State Through the Preparation-to-Delivery Transition",
        weight="bold",
    )
    for column, (prefix, title) in enumerate(
        (
            ("persistent_direction", "Persistent Channel Directions"),
            ("complete_role_reversal", "Complete Role Reversal"),
        )
    ):
        time_ms = 1000.0 * arrays[f"{prefix}__time_s"]
        target = (
            arrays[f"{prefix}__target_arm_torque_nm"]
            + arrays[f"{prefix}__target_wrist_torque_nm"]
        )
        transmitted = (
            arrays[f"{prefix}__transmitted_arm_torque_nm"]
            + arrays[f"{prefix}__transmitted_wrist_torque_nm"]
        )
        arm_activation = arrays[f"{prefix}__arm_activation"]
        wrist_activation = arrays[f"{prefix}__wrist_activation"]
        axes[0, column].plot(
            time_ms, target, color=INK, ls="--", lw=1.8, label="Desired Net"
        )
        axes[0, column].plot(
            time_ms, transmitted, color=DRIFT, lw=2.4, label="Transmitted Net"
        )
        axes[0, column].axvline(0.0, color=WARNING, lw=1.4)
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("Joint Moment (N m)")
        axes[0, column].legend(fontsize=7.5)
        axes[1, column].plot(
            time_ms, arm_activation[:, 0], color=TRANSFER, lw=2.1, label="Arm Positive"
        )
        axes[1, column].plot(
            time_ms,
            arm_activation[:, 1],
            color=TRANSFER,
            lw=1.5,
            ls=":",
            label="Arm Negative",
        )
        axes[1, column].plot(
            time_ms,
            wrist_activation[:, 0],
            color=DISTAL,
            lw=2.1,
            label="Wrist Positive",
        )
        axes[1, column].plot(
            time_ms,
            wrist_activation[:, 1],
            color=DISTAL,
            lw=1.5,
            ls=":",
            label="Wrist Negative",
        )
        axes[1, column].axvline(0.0, color=WARNING, lw=1.4)
        axes[1, column].set(
            xlabel="Time Relative to Transition (ms)", ylabel="Activation"
        )
        axes[1, column].legend(fontsize=7.0, ncol=2)
    fig.tight_layout()
    _save(fig, "fig_biological_role_reversal")


def make_engine_ladder(record: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.set_title(
        "Select the Engine From the Scientific Question, Then Compare Common Observables",
        weight="bold",
        pad=12,
    )
    engines = [
        ("MuJoCo", "Forward Contact", "$q, v$\nWrench • Power", DRIFT),
        ("Pinocchio", "Rigid-Body Dynamics", "$q, v, \\tau$\nJacobian", TRANSFER),
        (
            "Drake",
            "Constraints\nand Optimization",
            "$q, v, u$\nConstraint Wrench",
            WARNING,
        ),
        (
            "OpenSim",
            "Subject-Scaled\nMuscles",
            "Activation • Force\nMoment Arm • $\\tau$",
            DISTAL,
        ),
        (
            "MyoSuite",
            "Activation-Driven\nControl",
            "Excitation • Force\nContact Wrench",
            CONTROL,
        ),
    ]
    for index, (name, role, observables, color) in enumerate(engines):
        x = 0.25 + 2.15 * index
        box = FancyBboxPatch(
            (x, 2.1),
            1.85,
            1.35,
            boxstyle="round,pad=0.08",
            fc="white",
            ec=color,
            lw=2.2,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.925, 3.18, name, ha="center", weight="bold", color=color, fontsize=11
        )
        ax.text(x + 0.925, 2.74, role, ha="center", va="center", fontsize=8.0)
        ax.text(
            x + 0.925,
            2.30,
            observables,
            ha="center",
            va="center",
            fontsize=6.8,
            color=MUTED,
        )
        ax.plot([x + 0.925, 5.5], [2.1, 1.2], color=color, lw=1.4, alpha=0.65)
    common = FancyBboxPatch(
        (2.0, 0.35), 7.0, 0.9, boxstyle="round,pad=0.08", fc=INK, ec=INK
    )
    ax.add_patch(common)
    ax.text(
        5.5,
        0.93,
        "Canonical Comparison Surface",
        ha="center",
        color="white",
        weight="bold",
        fontsize=12,
    )
    ax.text(
        5.5,
        0.58,
        "Frames • Events • Wrenches • Generalized Moments • Power • Residuals",
        ha="center",
        color="white",
        fontsize=9,
    )
    ax.text(
        5.5,
        4.32,
        "Engine Agreement Tests Shared Assumptions; It Does Not Replace Human Validation",
        ha="center",
        color=CONTROL,
        weight="bold",
    )
    _save(fig, "fig_cross_engine_question_ladder")


def _draw_pose(ax: plt.Axes, phase: float, label: str) -> None:
    pelvis = np.array([0.0, 0.0])
    shoulder = pelvis + np.array([0.0, 0.78])
    rotation = -1.10 + 2.15 * phase
    lead_elbow = shoulder + 0.48 * np.array([np.cos(rotation), np.sin(rotation)])
    trail_elbow = shoulder + 0.43 * np.array(
        [np.cos(rotation - 0.28), np.sin(rotation - 0.28)]
    )
    hands = 0.5 * (lead_elbow + trail_elbow) + 0.42 * np.array(
        [np.cos(rotation + 0.30), np.sin(rotation + 0.30)]
    )
    club_angle = rotation + 1.45 - 0.95 * phase
    head = hands + 0.92 * np.array([np.cos(club_angle), np.sin(club_angle)])
    ax.plot(
        [pelvis[0], shoulder[0]], [pelvis[1], shoulder[1]], color=INK, lw=6, alpha=0.65
    )
    ax.plot(
        [shoulder[0], lead_elbow[0], hands[0]],
        [shoulder[1], lead_elbow[1], hands[1]],
        color=TRANSFER,
        lw=4.5,
    )
    ax.plot(
        [shoulder[0], trail_elbow[0], hands[0]],
        [shoulder[1], trail_elbow[1], hands[1]],
        color=DRIFT,
        lw=4.5,
    )
    ax.plot([hands[0], head[0]], [hands[1], head[1]], color=INK, lw=3.0)
    ax.scatter(*hands, color=WARNING, s=45, zorder=5)
    ax.scatter(*head, color=CONTROL, s=65, zorder=5)
    tangent = np.array([-np.sin(club_angle), np.cos(club_angle)])
    _arrow(ax, hands, hands + 0.42 * tangent, DISTAL, "$\\mathbf{F}_t$")
    ax.text(0.0, -0.28, label, ha="center", weight="bold")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.4, 1.72)
    ax.set_aspect("equal")
    ax.axis("off")


def make_motion_plate() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.3))
    fig.suptitle(
        "Phase-Resolved Geometry Makes the Transfer Mechanism Visible", weight="bold"
    )
    for ax, phase, label in zip(
        axes,
        (0.05, 0.36, 0.68, 0.94),
        ("Preparation", "Retention", "Rapid Distal Acceleration", "Delivery"),
        strict=True,
    ):
        _draw_pose(ax, phase, label)
    fig.tight_layout()
    _save(fig, "fig_advanced_model_motion_plate")


def main() -> None:
    """Generate all advanced bridge figures."""
    _style()
    record = json.loads((DATA_DIR / "advanced_biological_bridge.json").read_text())
    with np.load(DATA_DIR / "advanced_biological_bridge.npz") as arrays:
        make_frame_power_invariance()
        make_biological_redundancy(arrays)
        make_biological_role_reversal(arrays)
    make_engine_ladder(record)
    make_motion_plate()


if __name__ == "__main__":
    main()
