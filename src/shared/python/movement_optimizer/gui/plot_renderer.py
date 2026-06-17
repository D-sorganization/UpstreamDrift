# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Static plot renderers for the Movement Optimizer."""

from typing import Any

import numpy as np

from ..models import BodyModel
from ..models.chain_forces import ChainForceHistory
from ..models.swingset import SWING_POLICY_JOINT_NAMES
from ..models.swingset_forces import SwingForceHistory
from ..rendering import Palette, get_chart_color
from ..spine_loads import NIOSH_COMPRESSION_LIMIT, spinal_compression, spinal_shear
from ..trajectory import OptimizationResult

# Per-joint colours for swingset plots (5 joints; shared accessible cycle).
_JOINT_COLORS = tuple(get_chart_color(i) for i in range(len(SWING_POLICY_JOINT_NAMES)))


def plot_angles(ax: Any, r: OptimizationResult, labels: tuple = Palette.SEG_LABELS) -> None:
    n_dof = min(r.q.shape[1], len(labels))
    for j in range(n_dof):
        ax.plot(
            r.t,
            np.degrees(r.q[:, j]),
            color=Palette.SEG_COLORS[j % len(Palette.SEG_COLORS)],
            lw=2,
            label=labels[j],
        )
    ax.set_xlabel("Time (s)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel("Angle (deg)", color=Palette.FG_DIM, fontsize=8)
    ax.set_title("Joint Angles", color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=7,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )


def plot_torques(ax: Any, r: OptimizationResult, labels: tuple = Palette.SEG_LABELS) -> None:
    n_dof = min(r.torques.shape[1], len(labels))
    for j in range(n_dof):
        ax.plot(
            r.t,
            r.torques[:, j],
            color=Palette.SEG_COLORS[j % len(Palette.SEG_COLORS)],
            lw=2,
            label=labels[j],
        )
    ax.axhline(0, color=Palette.FG_DIM, lw=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel("Torque (N\u00b7m)", color=Palette.FG_DIM, fontsize=8)
    ax.set_title("Joint Torques", color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=7,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )


def plot_power(ax: Any, r: OptimizationResult, labels: tuple = Palette.SEG_LABELS) -> None:
    n_dof = min(r.power.shape[1], len(labels))
    for j in range(n_dof):
        ax.plot(
            r.t,
            r.power[:, j],
            color=Palette.SEG_COLORS[j % len(Palette.SEG_COLORS)],
            lw=2,
            label=labels[j],
        )
    ax.plot(
        r.t,
        np.sum(r.power, axis=1),
        "--",
        color=Palette.FG,
        lw=2,
        label="Total",
        alpha=0.7,
    )
    ax.axhline(0, color=Palette.FG_DIM, lw=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel("Power (W)", color=Palette.FG_DIM, fontsize=8)
    ax.set_title("Joint Power", color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=7,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )


def plot_com_path(ax: Any, r: OptimizationResult, body: BodyModel) -> None:
    import matplotlib as mpl

    # matplotlib>=3.7 (see pyproject) always provides the registry-based
    # ``colormaps`` accessor; ``cm.get_cmap`` was removed in 3.9.
    cmap = mpl.colormaps["viridis"]
    colors_t = cmap(np.linspace(0.2, 0.95, len(r.t)))
    for i in range(len(r.t) - 1):
        ax.plot(
            r.com[i : i + 2, 0] * 100,
            r.com[i : i + 2, 1] * 100,
            color=colors_t[i],
            lw=2.5,
        )
    ax.plot(
        r.bar[:, 0] * 100,
        r.bar[:, 1] * 100,
        "-",
        color=Palette.ORANGE,
        lw=1.5,
        alpha=0.7,
        label="Bar path",
    )
    ax.plot(
        [r.com[0, 0] * 100, r.com[-1, 0] * 100],
        [r.com[0, 1] * 100, r.com[-1, 1] * 100],
        "--",
        color=Palette.YELLOW,
        lw=1.2,
        alpha=0.5,
        label="COM straight",
    )
    ax.plot(
        r.com[0, 0] * 100,
        r.com[0, 1] * 100,
        "o",
        color=Palette.RED,
        ms=8,
        label="Start",
    )
    ax.plot(
        r.com[-1, 0] * 100,
        r.com[-1, 1] * 100,
        "s",
        color=Palette.GREEN,
        ms=8,
        label="End",
    )
    ax.axvline(body.inner_heel * 100, color=Palette.GREEN, ls="-", lw=1.2, alpha=0.7)
    ax.axvline(body.inner_toe * 100, color=Palette.GREEN, ls="-", lw=1.2, alpha=0.7)
    ax.axvline(body.heel_x * 100, color=Palette.ORANGE, ls=":", lw=1, alpha=0.4)
    ax.axvline(body.toe_x * 100, color=Palette.ORANGE, ls=":", lw=1, alpha=0.4)
    ax.set_xlabel("Horizontal (cm)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel("Height (cm)", color=Palette.FG_DIM, fontsize=8)
    ax.set_title("COM & Bar Path", color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=6,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )


def plot_com_balance(ax: Any, r: OptimizationResult, body: BodyModel) -> None:
    ax.plot(r.t, r.com[:, 0] * 100, color=Palette.ACCENT, lw=2, label="COM x")
    ax.axhline(body.inner_heel * 100, color=Palette.GREEN, ls="-", lw=1.5, alpha=0.8)
    ax.axhline(body.inner_toe * 100, color=Palette.GREEN, ls="-", lw=1.5, alpha=0.8)
    ax.fill_between(
        r.t,
        body.inner_heel * 100,
        body.inner_toe * 100,
        alpha=0.12,
        color=Palette.GREEN,
        label="Inner BOS (60%)",
    )
    ax.axhline(body.heel_x * 100, color=Palette.ORANGE, ls=":", lw=1, alpha=0.5)
    ax.axhline(body.toe_x * 100, color=Palette.ORANGE, ls=":", lw=1, alpha=0.5)
    ax.fill_between(
        r.t,
        body.heel_x * 100,
        body.toe_x * 100,
        alpha=0.04,
        color=Palette.ORANGE,
        label="Full BOS",
    )
    ax.axhline(body.inner_center * 100, color=Palette.GREEN, ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Time (s)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel("COM x (cm)", color=Palette.FG_DIM, fontsize=8)
    ax.set_title("COM Balance", color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=6,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )


def plot_spine_loads(
    ax_comp: Any, ax_shear: Any, r: OptimizationResult, body: BodyModel, bar_mass: float, name: str
) -> None:
    exercise_type = name.lower().replace(" ", "_")
    if exercise_type == "bottoms_up_squat":
        exercise_type = "squat"

    comp = spinal_compression(r.q, r.qd, r.qdd, body, bar_mass, exercise_type)
    shear = spinal_shear(r.q, r.qd, r.qdd, body, bar_mass, exercise_type)

    ax = ax_comp
    ax.plot(r.t, comp, color=Palette.RED, lw=2, label="L5/S1 compression")
    ax.axhline(
        NIOSH_COMPRESSION_LIMIT,
        color=Palette.YELLOW,
        ls="--",
        lw=1.5,
        alpha=0.8,
        label=f"NIOSH limit ({NIOSH_COMPRESSION_LIMIT:.0f} N)",
    )
    ax.fill_between(
        r.t,
        NIOSH_COMPRESSION_LIMIT,
        comp,
        where=comp > NIOSH_COMPRESSION_LIMIT,  # type: ignore[arg-type]
        alpha=0.3,
        color=Palette.RED,
        label="Exceeds limit",
    )
    ax.set_xlabel("Time (s)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel("Force (N)", color=Palette.FG_DIM, fontsize=8)
    ax.set_title("Spinal Compression (L5/S1)", color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=6,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )

    ax = ax_shear
    ax.plot(r.t, shear, color=Palette.ORANGE, lw=2, label="L5/S1 shear")
    ax.axhline(0, color=Palette.FG_DIM, lw=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel("Force (N)", color=Palette.FG_DIM, fontsize=8)
    ax.set_title("Spinal Shear (L5/S1)", color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=6,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )


# ---------------------------------------------------------------------------
# Swingset / chain analysis plots
# ---------------------------------------------------------------------------


def _style_timeseries_axis(ax: Any, ylabel: str, title: str, *, legend_fontsize: int = 7) -> None:
    """Apply the shared axis labels, title, and legend styling (DRY)."""
    ax.set_xlabel("Time (s)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel(ylabel, color=Palette.FG_DIM, fontsize=8)
    ax.set_title(title, color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=legend_fontsize,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )


def plot_swing_joint_torques(ax: Any, history: SwingForceHistory) -> None:
    for j, name in enumerate(SWING_POLICY_JOINT_NAMES):
        ax.plot(
            history.time_s,
            history.joint_torque_nm[:, j],
            color=_JOINT_COLORS[j],
            lw=1.8,
            label=name,
        )
    ax.axhline(0, color=Palette.FG_DIM, lw=0.5, alpha=0.3)
    _style_timeseries_axis(ax, "Torque (N·m)", "Joint Torques")


def plot_swing_joint_power(ax: Any, history: SwingForceHistory) -> None:
    for j, name in enumerate(SWING_POLICY_JOINT_NAMES):
        ax.plot(
            history.time_s,
            history.joint_power_w[:, j],
            color=_JOINT_COLORS[j],
            lw=1.8,
            label=name,
        )
    ax.plot(
        history.time_s,
        np.sum(history.joint_power_w, axis=1),
        "--",
        color=Palette.FG,
        lw=2,
        alpha=0.7,
        label="Total",
    )
    ax.axhline(0, color=Palette.FG_DIM, lw=0.5, alpha=0.3)
    _style_timeseries_axis(ax, "Power (W)", "Joint Power")


def plot_swing_angle(ax: Any, history: SwingForceHistory) -> None:
    ax.plot(
        history.time_s,
        np.degrees(history.swing_angle_rad),
        color=Palette.ACCENT,
        lw=2,
        label="Swing angle",
    )
    ax.axhline(0, color=Palette.FG_DIM, lw=0.5, alpha=0.3)
    _style_timeseries_axis(ax, "Angle (deg)", "Swing Angle")


def plot_swing_com_height(ax: Any, history: SwingForceHistory) -> None:
    ax.plot(history.time_s, history.com_height_m, color=Palette.GREEN, lw=2, label="COM height")
    _style_timeseries_axis(ax, "Height (m)", "COM Height")


def plot_swing_energy(ax: Any, history: SwingForceHistory) -> None:
    ax.plot(history.time_s, history.energy_j, color=Palette.ORANGE, lw=2, label="Swing energy")
    _style_timeseries_axis(ax, "Energy (J)", "Swing Energy")


def plot_swing_com_path(ax: Any, history: SwingForceHistory) -> None:
    # com_path_m is (x, +y-down); negate y so "up" is up on the plot.
    xs = history.com_path_m[:, 0]
    ys = -history.com_path_m[:, 1]
    ax.plot(xs, ys, color=Palette.BLUE, lw=2, label="COM path")
    ax.plot(xs[0], ys[0], "o", color=Palette.RED, ms=7, label="Start")
    ax.plot(xs[-1], ys[-1], "s", color=Palette.GREEN, ms=7, label="End")
    ax.set_xlabel("Horizontal (m)", color=Palette.FG_DIM, fontsize=8)
    ax.set_ylabel("Vertical (m)", color=Palette.FG_DIM, fontsize=8)
    ax.set_title("COM Path", color=Palette.FG, fontsize=10)
    ax.legend(
        fontsize=6,
        facecolor=Palette.BG_PANEL,
        edgecolor=Palette.FG_DIM,
        labelcolor=Palette.FG,
    )


def plot_chain_tension(ax: Any, history: ChainForceHistory) -> None:
    ax.plot(
        history.time_s, history.max_tension_n, color=Palette.RED, lw=2, label="Max link tension"
    )
    mean_tension = (
        np.mean(history.link_tension_n, axis=1)
        if history.link_tension_n.shape[1]
        else np.zeros_like(history.time_s)
    )
    ax.plot(
        history.time_s, mean_tension, color=Palette.ACCENT, lw=1.5, alpha=0.8, label="Mean tension"
    )
    _style_timeseries_axis(ax, "Tension (N)", "Chain Link Tension")


def plot_chain_curvature(ax: Any, history: ChainForceHistory) -> None:
    ax.plot(
        history.time_s,
        np.degrees(history.max_curvature_rad),
        color=Palette.ORANGE,
        lw=2,
        label="Max curvature",
    )
    _style_timeseries_axis(ax, "Curvature (deg)", "Chain Curvature")


def plot_chain_energy(ax: Any, time_s: Any, energy_j: Any) -> None:
    ax.plot(time_s, energy_j, color=Palette.GREEN, lw=2, label="Total energy")
    _style_timeseries_axis(ax, "Energy (J)", "Chain Energy")


def plot_chain_tip_speed(ax: Any, time_s: Any, tip_speed_m_s: Any) -> None:
    ax.plot(time_s, tip_speed_m_s, color=Palette.BLUE, lw=2, label="Tip speed")
    _style_timeseries_axis(ax, "Speed (m/s)", "Chain Tip Speed")
