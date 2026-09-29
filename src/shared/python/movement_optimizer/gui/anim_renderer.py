# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Animation/kinematic frame rendering for the Movement Optimizer."""

from typing import Any

import numpy as np
from matplotlib.patches import Circle as MplCircle

from ..constants import PLATE_RADIUS_STD_M
from ..models import BodyModel
from ..rendering import BarbellRenderer, BodyRenderer, Palette, style_axis
from ..trajectory import OptimizationResult


def draw_anim_frame(
    ax: Any,
    fi: int,
    result: OptimizationResult,
    dynamics: Any,
    body: BodyModel,
    exercise_name: str,
    exercise_type: str,
) -> None:
    n = len(result.t)
    fi = fi % n
    ax.clear()
    style_axis(ax)

    q = result.q[fi]
    t_now = result.t[fi]
    fk = dynamics.forward_kinematics(q)

    if exercise_type == "bench_press":
        _draw_bench_press_frame(ax, q, t_now, fi, result, fk, exercise_name)
    else:
        _draw_standing_frame(
            ax, q, t_now, fi, result, dynamics, body, fk, exercise_name, exercise_type
        )


# -- Bench press rendering constants ----------------------------
_BENCH_HEIGHT = 0.45
_BENCH_HEAD_X = -0.40
_BENCH_NECK_X = -0.30
_BENCH_SHOULDER_X = -0.18
_BENCH_HIP_X = 0.30

# Naturalistic illustration colours for the anatomical figure. These are
# physical-material colours (wood bench, skin, neutral hand) rather than UI
# chrome, so they are intentionally not themed.
_BENCH_WOOD = "#8B4513"
_SKIN_TONE = "#d4a574"
_HEAD_EDGE = "#333333"
_HAND_GREY = "#b0b0b0"


def _draw_bench_press_frame(
    ax: Any,
    q: Any,
    t_now: float,
    fi: int,
    result: OptimizationResult,
    fk: dict[str, Any],
    name: str,
) -> None:
    ax.set_xlim(-0.6, 0.8)
    ax.set_ylim(-0.15, 1.4)
    ax.set_aspect("equal", adjustable="datalim")

    _draw_bench_and_body(ax)
    hand_pos, arm_base, fk_origin = _draw_bench_arm_chain(ax, fk)
    _draw_bench_barbell_and_trace(ax, fi, result, hand_pos, arm_base, fk_origin)
    _draw_bench_title(ax, q, t_now, name)


def _draw_bench_and_body(ax: Any) -> None:
    bh = _BENCH_HEIGHT
    head_x = _BENCH_HEAD_X
    neck_x = _BENCH_NECK_X
    shoulder_x = _BENCH_SHOULDER_X
    hip_x = _BENCH_HIP_X

    # Bench surface
    ax.fill_between([-0.8, 0.5], bh - 0.05, bh, color=_BENCH_WOOD, alpha=0.6)

    # Neck
    ax.plot(
        [head_x + 0.08, neck_x],
        [bh + 0.05, bh + 0.02],
        "-",
        color=_SKIN_TONE,
        lw=5,
        solid_capstyle="round",
    )
    # Head
    ax.add_patch(
        MplCircle(
            (head_x, bh + 0.05),
            0.08,
            facecolor=_SKIN_TONE,
            edgecolor=_HEAD_EDGE,
            lw=1.2,
            alpha=0.85,
            zorder=6,
        )
    )
    # Torso (horizontal on bench)
    for x0, x1 in [(neck_x, shoulder_x), (shoulder_x, hip_x)]:
        ax.plot(
            [x0, x1],
            [bh + 0.02, bh + 0.02],
            "-",
            color=Palette.SEG_COLORS[2],
            lw=12,
            solid_capstyle="round",
        )
    # Legs hanging off bench to floor
    ax.plot(
        [hip_x, hip_x + 0.15],
        [bh, 0],
        "-",
        color=Palette.SEG_COLORS[1],
        lw=9,
        solid_capstyle="round",
    )
    ax.plot(
        [hip_x + 0.15, hip_x + 0.2],
        [0, 0],
        "-",
        color=Palette.SEG_COLORS[0],
        lw=6,
        solid_capstyle="round",
    )
    # Ground line
    ax.plot([-0.6, 0.8], [0, 0], color=Palette.FG_DIM, lw=2, alpha=0.3)


def _draw_bench_arm_chain(ax: Any, fk: dict[str, Any]) -> tuple[Any, Any, Any]:
    bh = _BENCH_HEIGHT
    shoulder_x = _BENCH_SHOULDER_X

    arm_base = np.array([shoulder_x, bh + 0.02])
    fk_points = [fk["ankle"], fk["knee"], fk["hip"], fk["shoulder"]]
    arm_joints = [arm_base + (pt - fk_points[0]) for pt in fk_points]

    # Arm segments: upper arm, forearm, hand/wrist
    colors = [Palette.SEG_COLORS[0], Palette.SEG_COLORS[1], _HAND_GREY]
    lws = [8, 6, 4]
    for k in range(3):
        ax.plot(
            [arm_joints[k][0], arm_joints[k + 1][0]],
            [arm_joints[k][1], arm_joints[k + 1][1]],
            "-",
            color=colors[k],
            lw=lws[k],
            solid_capstyle="round",
        )
    # Joint markers
    for pt in arm_joints:
        ax.plot(
            pt[0],
            pt[1],
            "o",
            color=Palette.FG,
            ms=6,
            markeredgecolor="#333",
            markeredgewidth=1,
        )
    return arm_joints[-1], arm_base, fk_points[0]


def _draw_bench_barbell_and_trace(
    ax: Any,
    fi: int,
    result: OptimizationResult,
    hand_pos: Any,
    arm_base: Any,
    fk_origin: Any,
) -> None:
    BarbellRenderer.draw(ax, (hand_pos[0], hand_pos[1]))

    bar_trace_offset = arm_base - fk_origin
    bench_bar_traj = result.bar + bar_trace_offset
    BodyRenderer.draw_bar_trace(ax, bench_bar_traj, fi)


def _draw_bench_title(ax: Any, q: Any, t_now: float, name: str) -> None:
    ax.set_title(
        f"{name}  t={t_now:.2f}s  |  "
        f"Shoulder {np.degrees(q[0]):.0f}\u00b0  "
        f"Elbow {np.degrees(q[1]):.0f}\u00b0  "
        f"Wrist {np.degrees(q[2]):.0f}\u00b0",
        color=Palette.FG,
        fontsize=10,
        fontweight="bold",
    )


def _draw_standing_frame(
    ax: Any,
    q: Any,
    t_now: float,
    fi: int,
    result: OptimizationResult,
    dynamics: Any,
    body: BodyModel,
    fk: dict[str, Any],
    name: str,
    exercise_type: str,
) -> None:
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.15, 1.8)
    ax.set_aspect("equal", adjustable="datalim")

    is_dl = exercise_type == "deadlift"

    BodyRenderer.draw_ground(ax, body.heel_x, body.toe_x)
    BodyRenderer.draw_ghost(ax, dynamics.forward_kinematics(result.q[0]))
    BodyRenderer.draw_ghost(ax, dynamics.forward_kinematics(result.q[-1]))
    BodyRenderer.draw_segments(ax, fk)

    shoulder = fk["shoulder"]
    if is_dl:
        BodyRenderer.draw_arms(ax, shoulder, body.L_arm)
        ax.axhline(PLATE_RADIUS_STD_M, color=Palette.FG_DIM, ls=":", lw=0.8, alpha=0.3)

    bp = dynamics.bar_position(q, exercise_type)
    bar_pos = (bp[0], bp[1])

    BarbellRenderer.draw(ax, bar_pos)
    BodyRenderer.draw_com_marker(ax, result.com[fi])
    BodyRenderer.draw_bar_trace(ax, result.bar, fi)

    ax.set_title(
        f"{name}  t={t_now:.2f}s  |  "
        f"Shin {np.degrees(q[0]):.0f}\u00b0  "
        f"Thigh {np.degrees(q[1]):.0f}\u00b0  "
        f"Torso {np.degrees(q[2]):.0f}\u00b0",
        color=Palette.FG,
        fontsize=10,
        fontweight="bold",
    )
