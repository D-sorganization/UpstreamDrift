"""Configuration for plotting module.

Provides PlotConfig and DEFAULT_CONFIG used by kinematics, energy, and other
plotting sub-modules to control figure layout, colour palette, line styling,
and grid behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Color scheme for professional plots
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "quaternary": "#d62728",
    "quinary": "#9467bd",
    "senary": "#8c564b",
    "accent": "#e377c2",
    "dark": "#7f7f7f",
    "grid": "#cccccc",
    "foreground": "#333333",
    "background": "#ffffff",
}

# Ordered color cycle for multi-series plots
COLOR_CYCLE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["tertiary"],
    COLORS["quaternary"],
    COLORS["quinary"],
    COLORS["senary"],
    COLORS["accent"],
    COLORS["dark"],
]


# ---------------------------------------------------------------------------
# ColorPalette helper
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColorPalette:
    """Immutable colour palette with named accessors and index-based cycling.

    Attributes:
        primary: Main plot colour.
        secondary: Second series colour.
        tertiary: Third series colour.
        quaternary: Fourth series colour.
        quinary: Fifth series colour.
        senary: Sixth series colour.
        accent: Accent / highlight colour.
        foreground: Axis / tick / label colour.
        background: Figure background colour.
        cycle: Ordered list for indexed access.
    """

    primary: str = COLORS["primary"]
    secondary: str = COLORS["secondary"]
    tertiary: str = COLORS["tertiary"]
    quaternary: str = COLORS["quaternary"]
    quinary: str = COLORS["quinary"]
    senary: str = COLORS["senary"]
    accent: str = COLORS["accent"]
    foreground: str = COLORS["foreground"]
    background: str = COLORS["background"]
    cycle: tuple[str, ...] = tuple(COLOR_CYCLE)

    def get_color(self, index: int) -> str:
        """Return a colour by index, wrapping around the cycle."""
        return self.cycle[index % len(self.cycle)]


# ---------------------------------------------------------------------------
# PlotConfig
# ---------------------------------------------------------------------------
@dataclass
class PlotConfig:
    """Configuration for creating and styling matplotlib figures.

    This is the single source of truth consumed by every plotting function
    in the *kinematics*, *energy*, *kinetics*, *club*, and other modules.

    Attributes:
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Dots per inch for raster output.
        line_width: Default line width.
        marker_size: Default marker/scatter size.
        title_size: Font size for titles.
        label_size: Font size for axis labels.
        tick_size: Font size for tick labels.
        legend_size: Font size for legend entries.
        show_grid: Whether to display a grid.
        grid_alpha: Grid line opacity.
        tight_layout: Apply ``tight_layout`` on figure creation.
        colors: Colour palette instance.
        style: Matplotlib style sheet name (applied on figure creation).
    """

    width: float = 10.0
    height: float = 6.0
    dpi: int = 100
    line_width: float = 1.5
    marker_size: float = 20.0
    title_size: float = 14.0
    label_size: float = 12.0
    tick_size: float = 10.0
    legend_size: float = 10.0
    show_grid: bool = True
    grid_alpha: float = 0.3
    tight_layout: bool = True
    colors: ColorPalette = field(default_factory=ColorPalette)
    style: str | None = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    def create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        **subplot_kw: Any,
    ) -> tuple[Figure, Any]:
        """Create a pre-styled matplotlib Figure + Axes.

        Args:
            nrows: Number of subplot rows.
            ncols: Number of subplot columns.
            **subplot_kw: Forwarded to ``plt.subplots``.

        Returns:
            ``(fig, ax)`` tuple. *ax* is a single ``Axes`` when
            ``nrows == ncols == 1``, otherwise a numpy array of axes.
        """
        if self.style:
            plt.style.use(self.style)

        fig, ax = plt.subplots(
            nrows,
            ncols,
            figsize=(self.width, self.height),
            dpi=self.dpi,
            **subplot_kw,
        )

        if self.tight_layout:
            fig.set_tight_layout(True)

        return fig, ax

    def apply_to_axes(self, ax: Any) -> None:
        """Apply common styling to an existing Axes object."""
        ax.tick_params(labelsize=self.tick_size)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha)

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------
    @classmethod
    def presentation(cls) -> PlotConfig:
        """Large, high-contrast config for slides / posters."""
        return cls(
            width=14.0,
            height=8.0,
            dpi=150,
            line_width=2.5,
            marker_size=40.0,
            title_size=20.0,
            label_size=16.0,
            tick_size=14.0,
            legend_size=14.0,
        )

    @classmethod
    def publication(cls) -> PlotConfig:
        """Compact, high-DPI config suitable for journal figures."""
        return cls(
            width=7.0,
            height=4.5,
            dpi=300,
            line_width=1.0,
            marker_size=12.0,
            title_size=11.0,
            label_size=10.0,
            tick_size=9.0,
            legend_size=9.0,
        )

    @classmethod
    def dashboard(cls) -> PlotConfig:
        """Config for multi-panel real-time dashboards."""
        return cls(
            width=5.0,
            height=3.5,
            dpi=100,
            line_width=1.2,
            marker_size=10.0,
            title_size=11.0,
            label_size=9.0,
            tick_size=8.0,
            legend_size=8.0,
            tight_layout=True,
        )


# ---------------------------------------------------------------------------
# Module-level default
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = PlotConfig()
