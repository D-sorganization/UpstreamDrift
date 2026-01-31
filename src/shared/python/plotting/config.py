"""Plotting Configuration Module.

Centralized configuration for all plotting operations including:
- Color schemes (light/dark/custom)
- Default plot dimensions
- Axis styling
- Font settings

This module ensures consistent visual appearance across all plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt


@dataclass
class ColorScheme:
    """Color scheme for plots.

    Defines a complete color palette for professional visualizations.
    Supports both light and dark themes.

    Attributes:
        primary: Main color for primary data series
        secondary: Color for secondary data
        tertiary: Color for tertiary data
        accent: Highlight color
        background: Plot background
        foreground: Text and axis color
        grid: Grid line color
        series: List of colors for multi-series plots
    """

    primary: str = "#1f77b4"
    secondary: str = "#ff7f0e"
    tertiary: str = "#2ca02c"
    quaternary: str = "#d62728"
    quinary: str = "#9467bd"
    senary: str = "#8c564b"
    accent: str = "#e377c2"
    background: str = "#ffffff"
    foreground: str = "#333333"
    grid: str = "#cccccc"
    series: list[str] = field(
        default_factory=lambda: [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )

    @classmethod
    def light(cls) -> "ColorScheme":
        """Create light theme color scheme."""
        return cls(
            background="#ffffff",
            foreground="#333333",
            grid="#e5e5e5",
        )

    @classmethod
    def dark(cls) -> "ColorScheme":
        """Create dark theme color scheme."""
        return cls(
            primary="#0A84FF",
            secondary="#FF9F0A",
            tertiary="#30D158",
            quaternary="#FF375F",
            quinary="#BF5AF2",
            senary="#AC8E68",
            accent="#64D2FF",
            background="#1c1c1e",
            foreground="#ffffff",
            grid="#3a3a3c",
            series=[
                "#0A84FF",
                "#30D158",
                "#FF9F0A",
                "#FF375F",
                "#BF5AF2",
                "#64D2FF",
                "#FFD60A",
                "#AC8E68",
            ],
        )

    def get_color(self, index: int) -> str:
        """Get a series color by index (wraps around)."""
        return self.series[index % len(self.series)]

    def apply_to_matplotlib(self) -> None:
        """Apply this color scheme to matplotlib defaults."""
        plt.rcParams["axes.facecolor"] = self.background
        plt.rcParams["figure.facecolor"] = self.background
        plt.rcParams["text.color"] = self.foreground
        plt.rcParams["axes.edgecolor"] = self.foreground
        plt.rcParams["axes.labelcolor"] = self.foreground
        plt.rcParams["xtick.color"] = self.foreground
        plt.rcParams["ytick.color"] = self.foreground
        plt.rcParams["grid.color"] = self.grid
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self.series)


@dataclass
class PlotConfig:
    """Configuration for plot appearance.

    Centralizes all plot styling options for consistency.

    Attributes:
        width: Default figure width in inches
        height: Default figure height in inches
        dpi: Dots per inch
        colors: Color scheme to use
        font_family: Font family for text
        title_size: Title font size
        label_size: Axis label font size
        tick_size: Tick label font size
        legend_size: Legend font size
        line_width: Default line width
        marker_size: Default marker size
        grid_alpha: Grid line transparency
        show_grid: Whether to show grid
    """

    width: float = 10
    height: float = 6
    dpi: int = 100
    colors: ColorScheme = field(default_factory=ColorScheme)
    font_family: str = "sans-serif"
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    line_width: float = 1.5
    marker_size: float = 6.0
    grid_alpha: float = 0.3
    show_grid: bool = True

    @classmethod
    def default(cls) -> "PlotConfig":
        """Create default plot configuration."""
        return cls()

    @classmethod
    def publication(cls) -> "PlotConfig":
        """Create configuration suitable for publication.

        Higher DPI, larger fonts, thicker lines.
        """
        return cls(
            width=8,
            height=5,
            dpi=300,
            title_size=16,
            label_size=14,
            tick_size=12,
            legend_size=11,
            line_width=2.0,
            marker_size=8.0,
        )

    @classmethod
    def presentation(cls) -> "PlotConfig":
        """Create configuration for presentations.

        Larger elements for visibility at a distance.
        """
        return cls(
            width=12,
            height=7,
            dpi=150,
            title_size=20,
            label_size=16,
            tick_size=14,
            legend_size=14,
            line_width=2.5,
            marker_size=10.0,
        )

    def apply_to_matplotlib(self) -> None:
        """Apply this configuration to matplotlib defaults."""
        self.colors.apply_to_matplotlib()

        plt.rcParams["figure.figsize"] = [self.width, self.height]
        plt.rcParams["figure.dpi"] = self.dpi
        plt.rcParams["font.family"] = self.font_family
        plt.rcParams["font.size"] = self.label_size
        plt.rcParams["axes.titlesize"] = self.title_size
        plt.rcParams["axes.labelsize"] = self.label_size
        plt.rcParams["xtick.labelsize"] = self.tick_size
        plt.rcParams["ytick.labelsize"] = self.tick_size
        plt.rcParams["legend.fontsize"] = self.legend_size
        plt.rcParams["lines.linewidth"] = self.line_width
        plt.rcParams["lines.markersize"] = self.marker_size
        plt.rcParams["axes.grid"] = self.show_grid
        plt.rcParams["grid.alpha"] = self.grid_alpha

    def create_figure(self, **kwargs: Any) -> tuple:
        """Create a figure with this configuration.

        Args:
            **kwargs: Additional arguments to plt.subplots

        Returns:
            Tuple of (figure, axes)
        """
        figsize = kwargs.pop("figsize", (self.width, self.height))
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi, **kwargs)

        # Apply styling
        fig.set_facecolor(self.colors.background)
        if hasattr(ax, "set_facecolor"):
            ax.set_facecolor(self.colors.background)
            ax.tick_params(colors=self.colors.foreground)
            ax.xaxis.label.set_color(self.colors.foreground)
            ax.yaxis.label.set_color(self.colors.foreground)
            ax.title.set_color(self.colors.foreground)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.colors.foreground)
            ax.grid(self.show_grid, alpha=self.grid_alpha, color=self.colors.grid)

        return fig, ax


# Default configurations
DEFAULT_CONFIG = PlotConfig.default()
PUBLICATION_CONFIG = PlotConfig.publication()
PRESENTATION_CONFIG = PlotConfig.presentation()

# Light and dark themes
LIGHT_THEME = PlotConfig(colors=ColorScheme.light())
DARK_THEME = PlotConfig(colors=ColorScheme.dark())

__all__ = [
    "ColorScheme",
    "PlotConfig",
    "DEFAULT_CONFIG",
    "PUBLICATION_CONFIG",
    "PRESENTATION_CONFIG",
    "LIGHT_THEME",
    "DARK_THEME",
]
