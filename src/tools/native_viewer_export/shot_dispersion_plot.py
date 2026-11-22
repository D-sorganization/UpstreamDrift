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
    ("Side (m, + right)", "side_m", "{:+.1f}"),
    ("Face angle (deg)", "face_angle_deg", "{:+.2f}"),
    ("Club path (deg)", "club_path_deg", "{:+.2f}"),
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
            label=f"{cfg.ellipse_sigma:g}-sigma ellipse",
        )
    )


def _pad(low: float, high: float, fraction: float, floor: float) -> tuple[float, float]:
    pad = max((high - low) * fraction, floor)
    return low - pad, high + pad


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
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.0), height_ratios=(1.3, 1.0))
    plan = fig.add_subplot(grid[:, 0])
    side = fig.add_subplot(grid[0, 1])
    table = fig.add_subplot(grid[1, 1])
    cmap = plt.get_cmap(plot["colormap"])
    for i, shot in enumerate(shots):
        trace, colour = shot.trace, cmap(i % cmap.N)
        plan.plot(trace[:, 1], trace[:, 0], color=colour, lw=plot["trace_linewidth"])
        plan.scatter(
            trace[-1, 1],
            trace[-1, 0],
            s=plot["landing_marker_size"],
            color=colour,
            edgecolor=plot["landing_marker_edge"],
            zorder=3,
        )
        side.plot(trace[:, 0], trace[:, 2], color=colour, lw=plot["trace_linewidth"])
    _draw_ellipse(plan, result, cfg)
    plan.axvline(0.0, color=plot["target_line_color"], ls="--", lw=1.0)
    plan.scatter([0.0], [0.0], s=plot["tee_marker_size"], c="black", marker="^")
    pts = result.landing_points()
    plan.set_xlim(*_pad(min(pts[:, 0].min(), 0), max(pts[:, 0].max(), 0), 0.5, 1.0))
    plan.set_ylim(*_pad(0.0, pts[:, 1].max(), plot["margin_fraction"], 1.0))
    plan.set_xlabel("Side (m), right of target line is positive")
    plan.set_ylabel("Forward (m)")
    plan.set_title("Plan View, From Behind the Ball (Lateral Scale Exaggerated)")
    side.set_xlabel("Forward (m)")
    side.set_ylabel("Height (m)")
    side.set_title("Side Profile")
    for ax in (plan, side):
        ax.grid(alpha=plot["grid_alpha"])
    plan.legend(loc="lower right")
    table.axis("off")
    shown = table.table(
        cellText=_table_cells(result),
        colLabels=["Metric", "Mean", "Std"],
        loc="center",
        cellLoc="center",
    )
    shown.auto_set_font_size(False)
    shown.set_fontsize(plot["table_font_size"])
    shown.scale(1.0, plot["table_row_scale"])
    table.set_title(f"Summary Of {len(shots)} Of {len(result.outcomes)} Shots")
    fig.suptitle(plot["title"], fontsize=plot["title_font_size"])
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=plot["dpi"])
    plt.close(fig)
    return path
