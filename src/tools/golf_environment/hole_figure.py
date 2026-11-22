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
from src.tools.native_viewer_export.display_units import (  # noqa: E402
    DisplayUnits,
    distance_text,
    load_display_units,
)


def side_text(side_m: float, units: DisplayUnits) -> str:
    """``"11 yd left"`` for a negative (right-positive) side; ``"on line"`` at zero.

    ``side_m`` is metres; the text is in ``units``.
    """
    shown = distance_text(abs(side_m), units)
    if float(shown.split()[0]) == 0.0:
        return "on line"
    return f"{shown} {'left' if side_m < 0 else 'right'}"


def axis_labels(units: DisplayUnits) -> tuple[str, str, str]:
    """``(lateral, height, down-range)`` axis labels naming the display unit."""
    u = units.distance_unit
    return (
        f"lateral ({u}): + left of line, - right",
        f"height ({u})",
        f"down-range from tee ({u}); lateral and height exaggerated",
    )


def _legend_label(shot: HoleShot, units: DisplayUnits) -> str:
    return (
        f"{shot.label}: carry {distance_text(shot.carry_m, units)}, "
        f"{side_text(shot.side_m, units)}"
    )


def _draw_hole(
    ax: Any, hole: Any, fig: dict[str, Any], length: float, scale: float
) -> None:
    """Rough, fairway, green, tee and pin in (down-range, left) hole axes.

    ``length`` is already in display units; ``scale`` converts the hole's metres.
    """
    half = hole.fairway_width * scale / 2.0
    ax.add_patch(
        Rectangle((0, -half), length, 2 * half, color=fig["fairway_rgb"], zorder=1)
    )
    ax.add_patch(
        Circle((length, 0), hole.green_radius * scale, color=fig["green_rgb"], zorder=2)
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
    units: DisplayUnits,
) -> None:
    tee, forward, left = hole_axes(hole)
    scale = units.distance_scale
    for shot in shots:
        rel = np.asarray(shot.path_m, dtype=float) - tee
        x, y, z = scale * (rel @ forward), scale * (rel @ left), scale * rel[:, 2]
        kw = {"color": shot.colour, "lw": fig["line_width"], "zorder": 5}
        plan.plot(x, y, label=_legend_label(shot, units), **kw)
        side.plot(x, z, **kw)
        _draw_roll(plan, side, shot, (x[-1], y[-1], z[-1]), fig, scale)
        dot = {"ms": fig["landing_marker_size"], "mfc": shot.colour, "mec": "white"}
        plan.plot(x[-1], y[-1], "o", zorder=7, **dot)
        side.plot(x[-1], z[-1], "o", zorder=7, **dot)


def _draw_roll(
    plan: Any,
    side: Any,
    shot: HoleShot,
    landing: tuple[float, float, float],
    fig: dict[str, Any],
    scale: float,
) -> None:
    """Roll as a ground segment landing to rest in the plan and the profile.

    Nothing is drawn for a shot without a roll. ``landing`` is (down-range,
    left, height) in display units; the roll is on the ground, so the profile
    keeps the landing height.
    """
    if shot.roll_offset_m is None:
        return
    x, y, z = landing
    end_x = x + scale * shot.roll_offset_m[0]
    end_y = y + scale * shot.roll_offset_m[1]
    kw = {"color": fig["roll_rgb"], "lw": fig["line_width"], "zorder": 5}
    plan.plot([x, end_x], [y, end_y], gid="roll", **kw)
    side.plot([x, end_x], [z, z], gid="roll", clip_on=False, **kw)
    rest = {"ms": fig["roll_marker_size"], "mfc": fig["roll_rgb"], "mec": shot.colour}
    plan.plot(end_x, end_y, "D", zorder=7, **rest)
    side.plot(end_x, z, "D", zorder=7, clip_on=False, **rest)


def _label_text(shot: HoleShot, units: DisplayUnits, spec: dict[str, Any]) -> str:
    """Carry and side, and roll and total when the shot has a roll."""
    shown = distance_text(abs(shot.side_m), units)
    if float(shown.split()[0]) == 0.0:
        side = spec["on_line"]
    else:
        side = f"{shown} {spec['side_left'] if shot.side_m < 0 else spec['side_right']}"
    values = {"carry": distance_text(shot.carry_m, units), "side": side}
    lines = spec["lines"][:1]
    if shot.roll_m is not None and shot.total_m is not None:
        values["roll"] = distance_text(shot.roll_m, units)
        values["total"] = distance_text(shot.total_m, units)
        lines = spec["lines"]
    return "\n".join(line.format(**values) for line in lines)


def _overlaps(a: Any, b: Any) -> bool:
    return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)


def _box_of(note: Any, renderer: Any, fig: dict[str, Any]) -> Any:
    """Pixel box of an annotation with its frame (the patch lags until drawn)."""
    frame = 0.25 * fig["label_font_size"] * note.figure.dpi / 72.0
    return note.get_window_extent(renderer).padded(frame)


def _place_label(
    plan: Any, anchor: tuple[float, float], text: str, fig: dict[str, Any], taken: list
) -> Any:
    """Annotate ``anchor`` at the first candidate offset that fits and is clear.

    A candidate fits when its box is inside the plan axes and overlaps neither
    the legend nor a box in ``taken``; when none does, the last candidate is
    shifted into the axes. The accepted box is appended to ``taken``.
    """
    renderer = plan.figure.canvas.get_renderer()
    pad = fig["label_edge_pad_px"]
    area = plan.get_window_extent(renderer).expanded(1.0, 1.0)
    inner = (area.x0 + pad, area.y0 + pad, area.x1 - pad, area.y1 - pad)
    box = {"boxstyle": "round,pad=0.25", "fc": "white", "alpha": fig["label_box_alpha"]}
    for dx, dy in fig["_label_offsets"]:
        note = plan.annotate(
            text,
            anchor,
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="top" if dy < -8 else ("bottom" if dy > 8 else "center"),
            fontsize=fig["label_font_size"],
            bbox=box,
            zorder=9,
            annotation_clip=False,
        )
        note.set_gid("landing_label")
        bb = _box_of(note, renderer, fig)
        inside = inner[0] <= bb.x0 and bb.x1 <= inner[2]
        inside = inside and inner[1] <= bb.y0 and bb.y1 <= inner[3]
        if inside and not any(_overlaps(bb, t) for t in taken):
            taken.append(bb)
            return note
        note.remove()
    note = plan.annotate(
        text,
        anchor,
        xytext=(dx, dy),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=fig["label_font_size"],
        bbox=box,
        zorder=9,
        annotation_clip=False,
    )
    note.set_gid("landing_label")
    taken.append(_box_of(note, renderer, fig))
    return note


def _label_landings(
    plan: Any,
    shots: Sequence[HoleShot],
    hole: Any,
    settings: dict[str, Any],
    units: DisplayUnits,
) -> None:
    """Label each shot at its rest point (landing without roll) on the plan."""
    spec = settings["landing_label"]
    fig = {**settings["figure"], "_label_offsets": spec["offsets_pt"]}
    tee, forward, left = hole_axes(hole)
    scale = units.distance_scale
    renderer = plan.figure.canvas.get_renderer()
    legend = plan.get_legend()
    taken = [] if legend is None else [legend.get_window_extent(renderer)]
    for shot in shots:
        rel = np.asarray(shot.path_m, dtype=float)[-1] - tee
        x, y = scale * float(rel @ forward), scale * float(rel @ left)
        if shot.roll_offset_m is not None:
            x, y = x + scale * shot.roll_offset_m[0], y + scale * shot.roll_offset_m[1]
        _place_label(plan, (x, y), _label_text(shot, units, spec), fig, taken)


def build_hole_figure(
    shots: Sequence[HoleShot],
    hole: Any,
    settings: dict[str, Any],
    units: DisplayUnits | None = None,
    label_at_landing: bool = False,
    subtitle: str | None = None,
) -> tuple[Any, Any, Any]:
    """The figure and its (plan, profile) axes; the caller saves and closes it.

    Preconditions: at least one shot (``ValueError``). With ``label_at_landing``
    each shot gets a box at its rest point stating carry, side, roll and total
    (the roll lines only for shots with a roll), inside the plan axes and clear
    of the legend and of each other. Shots with a roll also draw it on the
    ground in both views. ``subtitle`` (non-empty) is a second title line, e.g.
    the aim line. Without roll, label and subtitle the figure is as before.
    """
    if not shots:
        raise ValueError("at least one shot is required")
    if subtitle is not None and not subtitle.strip():
        raise ValueError("subtitle must not be empty")
    fig_s = settings["figure"]
    units = units or load_display_units()
    scale = units.distance_scale
    _, forward, left = hole_axes(hole)
    tee = np.asarray(hole.tee_position, dtype=float)
    length = scale * float(np.linalg.norm(np.asarray(hole.pin_position)[:2] - tee[:2]))
    plt.rcParams["font.size"] = fig_s["font_size"]
    fig, (plan, side) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(fig_s["width_in"], fig_s["height_in"]),
        dpi=fig_s["dpi"],
    )
    plan.set_facecolor(fig_s["rough_rgb"])
    _draw_hole(plan, hole, fig_s, length, scale)
    _draw_shots(plan, side, shots, hole, fig_s, units)
    sides = max(abs(s.side_m) for s in shots)
    rests = [s.side_m + s.roll_offset_m[1] for s in shots if s.roll_offset_m]
    sides = max([sides, *(abs(r) for r in rests)])
    lim = (
        scale * max(hole.fairway_width / 2.0, sides) * fig_s["lateral_margin_fraction"]
    )
    apex = scale * max(float(np.max(np.asarray(s.path_m)[:, 2])) for s in shots)
    plan.set_ylim(-lim, lim)
    side.set_ylim(0, max(apex, 1.0) * fig_s["height_margin_fraction"])
    reach = max(length, scale * max(s.total_m or s.carry_m for s in shots))
    plan.set_xlim(-0.02 * length, reach * fig_s["length_margin_fraction"])
    lateral, height, down_range = axis_labels(units)
    plan.set_ylabel(lateral)
    side.set_ylabel(height)
    side.set_xlabel(down_range)
    title = "Plan View, Tee on the Left, Golfer's Right Is Down"
    plan.set_title(
        title if subtitle is None else f"{title}\n{subtitle}",
        fontsize=fig_s["title_font_size"],
    )
    plan.legend(
        loc="upper center", fontsize=fig_s["font_size"], ncol=1, framealpha=0.9
    ).set_zorder(10)
    for ax in (plan, side):
        ax.grid(alpha=fig_s["grid_alpha"])
    fig.tight_layout()
    if label_at_landing:
        fig.canvas.draw()
        _label_landings(plan, shots, hole, settings, units)
    return fig, plan, side


def write_hole_figure(
    shots: Sequence[HoleShot],
    hole: Any,
    path: Path,
    settings: dict[str, Any],
    units: DisplayUnits | None = None,
    label_at_landing: bool = False,
    subtitle: str | None = None,
) -> Path:
    """Write a PNG with plan view (top) and side profile (bottom); return ``path``.

    Preconditions: at least one shot (``ValueError``). Postcondition: ``path``
    exists; its legend gives each shot's carry and side (left or right) and every
    axis is in ``units`` (default: the display-units settings, yards). See
    :func:`build_hole_figure` for the roll segment and ``label_at_landing``.
    """
    fig, _, _ = build_hole_figure(
        shots, hole, settings, units, label_at_landing, subtitle
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
