# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Matplotlib rendering helpers for the animation panel.

Separated from the GUI so rendering logic can be tested independently
and reused with different frontends.

Design Principles:
    DRY  -- common drawing patterns are factored into class methods.
    LoD  -- renderers accept only the data they need, not whole objects.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from numpy.typing import NDArray

from .constants import BAR_RADIUS_M, LENGTH_FRAC, PLATE_RADIUS_STD_M
from .theme_bridge import (
    BUILTIN_THEMES,
    SHARED_THEME_AVAILABLE,
    ThemedWindowMixin,
    apply_plot_theme,
    get_chart_color,
    get_theme_manager,
)

# Re-exported so the rest of the app imports theme helpers from one place; this
# module bridges to the fleet shared theme (``shared.python.theme``) when it is
# installed and falls back to bundled fleet colours otherwise.
__theme_reexports__ = (
    SHARED_THEME_AVAILABLE,
    BUILTIN_THEMES,
    ThemedWindowMixin,
    apply_plot_theme,
    get_chart_color,
    get_theme_manager,
)

# Neck length as a fraction of body height (used by BodyRenderer)
NECK_LENGTH_FRAC: float = LENGTH_FRAC["neck"]


# Semantic colours (success/error) are not present in every built-in theme's
# token set, so they fall back to fixed accessible values.
_FALLBACK_GREEN = "#4ec9b0"
_FALLBACK_RED = "#f44747"


class Palette:
    """Centralised colour definitions sourced from the fleet shared theme.

    Class attributes are seeded from the ``Dark`` built-in theme at import time
    (no ``QApplication`` required) and refreshed in place by
    :func:`refresh_palette` whenever the active theme changes.
    """

    BG = "#1a1d23"
    BG_PANEL = "#24272e"
    BG_INPUT = "#0d1117"
    BG_PLOT = "#24272e"
    FG = "#e1e4e8"
    FG_DIM = "#c9d1d9"
    ACCENT = "#4a7ba7"
    ACCENT2 = "#5a8fc4"
    GREEN = _FALLBACK_GREEN
    RED = _FALLBACK_RED
    BLUE = "#58a6ff"
    ORANGE = "#ffb74d"
    YELLOW = "#ffd54f"
    SEG_COLORS = ("#569cd6", "#f44747", "#4ec9b0")
    SEG_LABELS = ("Lower leg", "Upper leg", "Torso")
    BENCH_LABELS = ("Shoulder", "Elbow", "Wrist")


def _apply_palette_colors(colors: Mapping[str, str]) -> None:
    """Populate :class:`Palette` class attributes from a theme colour mapping.

    Preconditions:
        ``colors`` provides the core theme keys (``bg``, ``group_bg``,
        ``input_bg``, ``text``, ``text_secondary``, ``accent``). Raises
        ``KeyError`` via direct indexing if a required key is absent.
    """
    Palette.BG = colors["bg"]
    Palette.BG_PANEL = colors["group_bg"]
    Palette.BG_INPUT = colors["input_bg"]
    Palette.BG_PLOT = colors.get("group_bg", colors["bg"])
    Palette.FG = colors["text"]
    Palette.FG_DIM = colors["text_secondary"]
    Palette.ACCENT = colors["accent"]
    Palette.ACCENT2 = colors.get("button_hover", colors["accent"])
    Palette.GREEN = colors.get("success", _FALLBACK_GREEN)
    Palette.RED = colors.get("error", _FALLBACK_RED)
    Palette.BLUE = colors.get("focus", colors["accent"])
    Palette.ORANGE = get_chart_color(1)
    Palette.YELLOW = get_chart_color(2)
    Palette.SEG_COLORS = (get_chart_color(0), get_chart_color(1), get_chart_color(2))


def refresh_palette() -> None:
    """Refresh :class:`Palette` from the live theme manager's current colours."""
    _apply_palette_colors(get_theme_manager().get_current_colors())


def restyle_figure(fig: object) -> None:
    """Apply the active theme to a matplotlib figure.

    Preconditions:
        ``fig`` is a matplotlib ``Figure``. Raises ``ValueError`` when ``None``.
    """
    if fig is None:
        raise ValueError("fig must be a matplotlib Figure")
    apply_plot_theme(fig, get_theme_manager().get_current_colors())


# Seed Palette from the Dark theme at import time (QApplication-free).
_apply_palette_colors(BUILTIN_THEMES["Dark"])


class BarbellRenderer:
    """Draw barbell as seen from the side (sagittal plane).

    From the side, a loaded barbell appears as a large circle (the plate)
    with a smaller concentric circle (the bar shaft).
    """

    BAR_ALPHA = 0.60
    PLATE_ALPHA = 0.50
    BAR_COLOR = "#c0c0c0"
    BAR_EDGE = "#888888"
    PLATE_COLOR = "#444444"
    PLATE_EDGE = "#333333"

    @classmethod
    def draw(cls, ax: Axes, position: tuple[float, float]) -> None:
        x, y = position
        cls._draw_plate_circle(ax, x, y)
        cls._draw_bar_circle(ax, x, y)

    @classmethod
    def _draw_plate_circle(cls, ax: Axes, x: float, y: float) -> None:
        """Outer plate circle — the bumper plate seen from the side."""
        ax.add_patch(
            Circle(
                (x, y),
                PLATE_RADIUS_STD_M,
                facecolor=cls.PLATE_COLOR,
                edgecolor=cls.PLATE_EDGE,
                linewidth=1.5,
                alpha=cls.PLATE_ALPHA,
                zorder=7,
            )
        )

    @classmethod
    def _draw_bar_circle(cls, ax: Axes, x: float, y: float) -> None:
        """Inner bar shaft circle — the bar cross-section."""
        ax.add_patch(
            Circle(
                (x, y),
                BAR_RADIUS_M,
                facecolor=cls.BAR_COLOR,
                edgecolor=cls.BAR_EDGE,
                linewidth=1.5,
                alpha=cls.BAR_ALPHA,
                zorder=8,
            )
        )


class BodyRenderer:
    """Draw the stick-figure body on a matplotlib Axes."""

    @classmethod
    def draw_ground(cls, ax: Axes, heel_x: float, toe_x: float) -> None:
        ax.plot([-0.7, 0.7], [0, 0], color=Palette.FG_DIM, lw=2, alpha=0.3)
        ax.plot(
            [heel_x, toe_x],
            [0, 0],
            color=Palette.ORANGE,
            lw=4,
            solid_capstyle="round",
            alpha=0.5,
        )

    @classmethod
    def draw_ghost(
        cls,
        ax: Axes,
        joints: dict[str, NDArray],
        alpha: float = 0.10,
        body_height: float = 1.75,
    ) -> None:
        pts = [joints["ankle"], joints["knee"], joints["hip"], joints["shoulder"]]
        for k in range(3):
            ax.plot(
                [pts[k][0], pts[k + 1][0]],
                [pts[k][1], pts[k + 1][1]],
                "-",
                color=Palette.FG,
                lw=2,
                alpha=alpha,
            )
        # Ghost neck + head
        neck_length = NECK_LENGTH_FRAC * body_height
        shoulder = joints["shoulder"]
        neck_top = np.array([shoulder[0], shoulder[1] + neck_length])
        ax.plot(
            [shoulder[0], neck_top[0]],
            [shoulder[1], neck_top[1]],
            "-",
            color=Palette.FG,
            lw=2,
            alpha=alpha,
        )
        ax.add_patch(
            Circle(
                (neck_top[0], neck_top[1] + cls.HEAD_RADIUS),
                cls.HEAD_RADIUS,
                facecolor=Palette.FG,
                edgecolor="none",
                alpha=alpha,
                zorder=6,
            )
        )

    # Linewidths per segment: shank (thinner), thigh (medium), torso (thick)
    SEG_LINEWIDTHS = (6, 9, 12)
    HEAD_RADIUS = 0.10  # metres

    @classmethod
    def draw_segments(cls, ax: Axes, joints: dict[str, NDArray], body_height: float = 1.75) -> None:
        pts = [joints["ankle"], joints["knee"], joints["hip"], joints["shoulder"]]
        for k in range(3):
            ax.plot(
                [pts[k][0], pts[k + 1][0]],
                [pts[k][1], pts[k + 1][1]],
                "-",
                color=Palette.SEG_COLORS[k],
                lw=cls.SEG_LINEWIDTHS[k],
                solid_capstyle="round",
            )
        for pt in pts:
            ax.plot(
                pt[0],
                pt[1],
                "o",
                color=Palette.FG,
                ms=7,
                markeredgecolor="#333",
                markeredgewidth=1.2,
            )
        # Draw neck segment from shoulder upward
        neck_length = NECK_LENGTH_FRAC * body_height
        shoulder = joints["shoulder"]
        neck_top = np.array([shoulder[0], shoulder[1] + neck_length])
        ax.plot(
            [shoulder[0], neck_top[0]],
            [shoulder[1], neck_top[1]],
            "-",
            color="#d4a574",
            lw=5,
            solid_capstyle="round",
        )
        # Draw head at top of neck
        cls.draw_head(ax, neck_top)

    @classmethod
    def draw_head(cls, ax: Axes, neck_top: NDArray) -> None:
        """Draw a circle representing the head above the top of the neck."""
        head_center = (neck_top[0], neck_top[1] + cls.HEAD_RADIUS)
        ax.add_patch(
            Circle(
                head_center,
                cls.HEAD_RADIUS,
                facecolor="#d4a574",
                edgecolor="#333",
                linewidth=1.2,
                alpha=0.85,
                zorder=6,
            )
        )

    @classmethod
    def draw_arms(cls, ax: Axes, shoulder: NDArray, arm_length: float) -> None:
        hand_y = shoulder[1] - arm_length
        ax.plot(
            [shoulder[0], shoulder[0]],
            [shoulder[1], hand_y],
            "-",
            color="#b0b0b0",
            lw=3,
            solid_capstyle="round",
        )
        ax.plot(shoulder[0], hand_y, "o", color=Palette.FG, ms=5)

    @classmethod
    def draw_com_marker(cls, ax: Axes, com: NDArray) -> None:
        ax.plot(com[0], com[1], "+", color=Palette.YELLOW, ms=12, mew=2)
        ax.annotate(
            "COM",
            (com[0], com[1]),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=7,
            color=Palette.YELLOW,
        )

    @classmethod
    def draw_bar_trace(cls, ax: Axes, bar_traj: NDArray, current_idx: int) -> None:
        ax.plot(
            bar_traj[:, 0],
            bar_traj[:, 1],
            "-",
            color=Palette.ORANGE,
            lw=1,
            alpha=0.25,
        )
        ax.plot(
            bar_traj[current_idx, 0],
            bar_traj[current_idx, 1],
            "x",
            color=Palette.ORANGE,
            ms=6,
            mew=1.5,
            alpha=0.7,
        )


def style_axis(ax: Axes) -> None:
    """Apply the active theme's styling to a matplotlib Axes (from the Palette)."""
    ax.set_facecolor(Palette.BG_PLOT)
    ax.tick_params(colors=Palette.FG_DIM, which="both", labelsize=8)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(Palette.FG_DIM)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.12, color=Palette.FG_DIM)
