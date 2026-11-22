"""Headless plan-view and side-profile figure of shots on a hole (#15150).

The dependable offscreen output: matplotlib (Agg), no OpenGL. Plan view has the
tee on the left and the golfer's right downwards, so a draw is drawn above the
target line. Lateral and height axes are exaggerated against down-range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from src.tools.golf_environment.shots import HoleShot, hole_axes  # noqa: E402


def side_text(side_m: float, decimals: int) -> str:
    """``"11 m left"`` for a negative (right-positive) side; ``"on line"`` at zero."""
    size = round(abs(side_m), decimals)
    if size == 0:
        return "on line"
    return f"{size:.{decimals}f} m {'left' if side_m < 0 else 'right'}"


def _legend_label(shot: HoleShot, decimals: int) -> str:
    return (
        f"{shot.label}: carry {shot.carry_m:.{decimals}f} m, "
        f"{side_text(shot.side_m, decimals)}"
    )


def _draw_hole(ax: Any, hole: Any, fig: dict[str, Any], length: float) -> None:
    """Rough, fairway, green, tee and pin in (down-range, left) hole axes."""
    half = hole.fairway_width / 2.0
    ax.add_patch(
        Rectangle((0, -half), length, 2 * half, color=fig["fairway_rgb"], zorder=1)
    )
    ax.add_patch(
        Circle((length, 0), hole.green_radius, color=fig["green_rgb"], zorder=2)
    )
    ax.plot([0], [0], "s", ms=fig["tee_marker_size"], color=fig["tee_rgb"], zorder=6)
    ax.plot(
        [length], [0], "^", ms=fig["tee_marker_size"], color=fig["pin_rgb"], zorder=6
    )
    ax.axhline(0, color=fig["tee_rgb"], lw=0.8, ls=":", zorder=3)


def _draw_shots(
    plan: Any,
    side: Any,
    shots: Sequence[HoleShot],
    hole: Any,
    fig: dict[str, Any],
    decimals: int,
) -> None:
    tee, forward, left = hole_axes(hole)
    for shot in shots:
        rel = np.asarray(shot.path_m, dtype=float) - tee
        x, y, z = rel @ forward, rel @ left, rel[:, 2]
        kw = {"color": shot.colour, "lw": fig["line_width"], "zorder": 5}
        plan.plot(x, y, label=_legend_label(shot, decimals), **kw)
        side.plot(x, z, **kw)
        dot = {"ms": fig["landing_marker_size"], "mfc": shot.colour, "mec": "white"}
        plan.plot(x[-1], y[-1], "o", zorder=7, **dot)
        side.plot(x[-1], z[-1], "o", zorder=7, **dot)


def write_hole_figure(
    shots: Sequence[HoleShot], hole: Any, path: Path, settings: dict[str, Any]
) -> Path:
    """Write a PNG with plan view (top) and side profile (bottom); return ``path``.

    Preconditions: at least one shot (``ValueError``). Postcondition: ``path``
    exists; its legend gives each shot's carry and side (left or right).
    """
    if not shots:
        raise ValueError("at least one shot is required")
    fig_s, decimals = settings["figure"], int(settings["decimals"])
    _, forward, left = hole_axes(hole)
    tee = np.asarray(hole.tee_position, dtype=float)
    length = float(np.linalg.norm(np.asarray(hole.pin_position)[:2] - tee[:2]))
    plt.rcParams["font.size"] = fig_s["font_size"]
    fig, (plan, side) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(fig_s["width_in"], fig_s["height_in"]),
        dpi=fig_s["dpi"],
    )
    plan.set_facecolor(fig_s["rough_rgb"])
    _draw_hole(plan, hole, fig_s, length)
    _draw_shots(plan, side, shots, hole, fig_s, decimals)
    sides = max(abs(s.side_m) for s in shots)
    lim = max(hole.fairway_width / 2.0, sides) * fig_s["lateral_margin_fraction"]
    apex = max(float(np.max(np.asarray(s.path_m)[:, 2])) for s in shots)
    plan.set_ylim(-lim, lim)
    side.set_ylim(0, max(apex, 1.0) * fig_s["height_margin_fraction"])
    plan.set_xlim(-0.02 * length, length * fig_s["length_margin_fraction"])
    plan.set_ylabel("lateral (m): + left of line, - right")
    side.set_ylabel("height (m)")
    side.set_xlabel("down-range from tee (m); lateral and height exaggerated")
    plan.set_title(
        "Plan view, tee on the left, golfer's right is down",
        fontsize=fig_s["title_font_size"],
    )
    plan.legend(
        loc="upper center", fontsize=fig_s["font_size"], ncol=1, framealpha=0.9
    ).set_zorder(10)
    for ax in (plan, side):
        ax.grid(alpha=fig_s["grid_alpha"])
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
