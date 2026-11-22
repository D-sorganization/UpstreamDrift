"""Figure of a shot dispersion run: plan view, side profile and summary table.

Colours, sizes and resolution come from the ``plot`` block of
``shot_dispersion_settings.json``. Lateral is right-positive as seen from
behind the ball looking down the target line (so right is right on the page).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.tools.native_viewer_export.shot_dispersion import (
    DispersionResult,
    DispersionSettings,
    landing_ellipse,
)

logger = logging.getLogger(__name__)

_TABLE_ROWS = (
    ("Carry (m)", "carry_m", "{:.1f}"),
    ("Side (m, + Right)", "side_m", "{:+.1f}"),
    ("Face Angle (deg)", "face_angle_deg", "{:+.2f}"),
    ("Club Path (deg)", "club_path_deg", "{:+.2f}"),
    ("Launch (deg)", "launch_deg", "{:.1f}"),
    ("Spin (rpm)", "spin_rpm", "{:.0f}"),
)


def _table_cells(result: DispersionResult) -> list[list[str]]:
    summary = result.summary()
    cells = []
    for label, key, fmt in _TABLE_ROWS:
        stat = summary[key]
        if stat is None:
            cells.append([label, "n/a", "n/a"])
        else:
            cells.append([label, fmt.format(stat["mean"]), f"{stat['std']:.2f}"])
    return cells


def _draw_ellipse(ax: Any, result: DispersionResult, cfg: DispersionSettings) -> None:
    from matplotlib.patches import Ellipse

    plot = cfg.plot
    fit = landing_ellipse(
        result.landing_points(), cfg.ellipse_sigma, cfg.min_points_for_ellipse
    )
    if fit is None:
        return
    centre, width, height, angle = fit
    ax.add_patch(
        Ellipse(
            (float(centre[0]), float(centre[1])),
            width,
            height,
            angle=angle,
            fill=False,
            edgecolor=plot["ellipse_edge_color"],
            linewidth=plot["ellipse_linewidth"],
        )
    )


def _pad(low: float, high: float, fraction: float, floor: float) -> tuple[float, float]:
    pad = max((high - low) * fraction, floor)
    return low - pad, high + pad


def _landing_scatter(ax: Any, point: Any, colour: Any, plot: Any) -> None:
    ax.scatter(
        point[0],
        point[1],
        s=plot["landing_marker_size"],
        color=colour,
        edgecolor=plot["landing_marker_edge"],
        zorder=3,
    )


def _draw_traces(plan: Any, side: Any, shots: Any, plot: Any, cmap: Any) -> None:
    for i, shot in enumerate(shots):
        trace, colour = shot.trace, cmap(i % cmap.N)
        plan.plot(trace[:, 1], trace[:, 0], color=colour, lw=plot["trace_linewidth"])
        _landing_scatter(plan, (trace[-1, 1], trace[-1, 0]), colour, plot)
        side.plot(trace[:, 0], trace[:, 2], color=colour, lw=plot["trace_linewidth"])


def _draw_zoom(
    ax: Any, result: DispersionResult, cfg: DispersionSettings, cmap: Any
) -> None:
    """Landing points and the ellipse alone, at equal scale on both axes."""
    plot, pts = cfg.plot, result.landing_points()
    for i, point in enumerate(pts):
        _landing_scatter(ax, point, cmap(i % cmap.N), plot)
    _draw_ellipse(ax, result, cfg)
    spread = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    half = max(
        spread * (0.5 + plot["zoom_margin_fraction"]), plot["zoom_min_half_width_m"]
    )
    cx, cy = pts.mean(axis=0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axvline(0.0, color=plot["target_line_color"], ls="--", lw=1.0)
    ax.set_xlabel("Side (m, + Right)")
    ax.set_ylabel("Forward (m)")
    ax.set_title(plot["zoom_title"])


def _legend(ax: Any, cfg: DispersionSettings) -> None:
    from matplotlib.lines import Line2D

    plot = cfg.plot
    handle = Line2D(
        [0],
        [0],
        color=plot["ellipse_edge_color"],
        lw=plot["ellipse_linewidth"],
        label=f"{cfg.ellipse_sigma:g}-Sigma Ellipse",
    )
    ax.legend(handles=[handle], loc="lower right")


def _draw_table(ax: Any, result: DispersionResult, plot: Any) -> None:
    ax.axis("off")
    shown = ax.table(
        cellText=_table_cells(result),
        colLabels=["Metric", "Mean", "Std"],
        loc="center",
        cellLoc="center",
    )
    shown.auto_set_font_size(False)
    shown.set_fontsize(plot["table_font_size"])
    shown.scale(1.0, plot["table_row_scale"])
    ax.set_title(f"Summary of {len(result.available)} of {len(result.outcomes)} Shots")


def draw_dispersion(
    result: DispersionResult, cfg: DispersionSettings, path: Path
) -> Path:
    """Write the figure to ``path`` (PNG). Raises ``ValueError`` when no shot
    has a flight; ``ImportError`` without matplotlib. Returns ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    shots = result.available
    if not shots:
        raise ValueError("no shot has a flight to draw")
    plot = cfg.plot
    plt.rcParams["font.size"] = plot["font_size"]
    fig = plt.figure(figsize=tuple(plot["figsize_in"]), dpi=plot["dpi"])
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.0), height_ratios=(1, 1))
    plan = fig.add_subplot(grid[:, 0])
    side = fig.add_subplot(grid[0, 1:])
    zoom = fig.add_subplot(grid[1, 1])
    table = fig.add_subplot(grid[1, 2])
    cmap = plt.get_cmap(plot["colormap"])
    _draw_traces(plan, side, shots, plot, cmap)
    _draw_ellipse(plan, result, cfg)
    plan.axvline(0.0, color=plot["target_line_color"], ls="--", lw=1.0)
    plan.scatter([0.0], [0.0], s=plot["tee_marker_size"], c="black", marker="^")
    pts = result.landing_points()
    plan.set_xlim(*_pad(min(pts[:, 0].min(), 0), max(pts[:, 0].max(), 0), 0.5, 1.0))
    plan.set_ylim(*_pad(0.0, pts[:, 1].max(), plot["margin_fraction"], 1.0))
    plan.set_xlabel("Side (m), Right of Target Line Is Positive")
    plan.set_ylabel("Forward (m)")
    plan.set_title("Plan View from Behind the Ball (Lateral Scale Exaggerated)")
    side.set_xlabel("Forward (m)")
    side.set_ylabel("Height (m)")
    side.set_title("Side Profile")
    _draw_zoom(zoom, result, cfg, cmap)
    for ax in (plan, side, zoom):
        ax.grid(alpha=plot["grid_alpha"])
    _legend(plan, cfg)
    _draw_table(table, result, plot)
    fig.suptitle(plot["title"], fontsize=plot["title_font_size"])
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=plot["dpi"])
    plt.close(fig)
    return path
