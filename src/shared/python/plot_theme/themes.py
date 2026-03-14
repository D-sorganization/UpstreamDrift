"""Plot theme definitions for matplotlib visualizations.

This module defines color themes for scientific plots, providing consistent
styling across histograms, scatter plots, line plots, contours, and more.

The "Scientific Violet" theme is based on the reference visualization featuring:
- Purple/magenta primary colors for data
- Blue/cyan for fitted curves and secondary elements
- Green/yellow gradients for contours
- Light backgrounds with subtle gradients
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlotTheme:
    """Configuration for a matplotlib plot theme.

    Attributes:
        name: Human-readable theme name
        description: Brief description of the theme style

        # Background colors
        figure_facecolor: Background color for the figure
        axes_facecolor: Background color for the axes/plot area
        axes_edgecolor: Color of axes border

        # Text colors
        text_color: Default text color
        title_color: Color for plot titles
        label_color: Color for axis labels
        tick_color: Color for tick marks and labels

        # Grid
        grid_color: Color for grid lines
        grid_alpha: Transparency of grid lines
        grid_linestyle: Style of grid lines

        # Primary data colors (for main data series)
        primary_colors: List of colors for primary data (histograms, scatter, etc.)
        primary_color: Main primary color
        primary_alpha: Default alpha for primary data

        # Secondary colors (for fitted curves, overlays)
        secondary_colors: List of colors for secondary elements
        secondary_color: Main secondary color

        # Accent colors (for highlights, annotations)
        accent_colors: List of accent colors
        accent_color: Main accent color

        # Contour/heatmap colormap
        contour_cmap: Colormap name or list for contours
        heatmap_cmap: Colormap name or list for heatmaps

        # Line styles
        line_width: Default line width
        marker_size: Default marker size

        # Font settings
        font_family: Font family for text
        font_size: Base font size
        title_size: Font size for titles
        label_size: Font size for labels

        # Additional matplotlib rcParams
        rcparams: Additional rcParams to apply
    """

    name: str
    description: str = ""

    # Background colors
    figure_facecolor: str = "#ffffff"
    axes_facecolor: str = "#ffffff"
    axes_edgecolor: str = "#333333"

    # Text colors
    text_color: str = "#333333"
    title_color: str = "#222222"
    label_color: str = "#333333"
    tick_color: str = "#666666"

    # Grid
    grid_color: str = "#cccccc"
    grid_alpha: float = 0.5
    grid_linestyle: str = "-"

    # Primary data colors
    primary_colors: list[str] = field(
        default_factory=lambda: ["#8B5CF6", "#A78BFA", "#C4B5FD"]
    )
    primary_color: str = "#8B5CF6"
    primary_alpha: float = 0.8

    # Secondary colors
    secondary_colors: list[str] = field(
        default_factory=lambda: ["#3B82F6", "#60A5FA", "#93C5FD"]
    )
    secondary_color: str = "#3B82F6"

    # Accent colors
    accent_colors: list[str] = field(
        default_factory=lambda: ["#10B981", "#34D399", "#6EE7B7"]
    )
    accent_color: str = "#10B981"

    # Colormaps
    contour_cmap: str | list[str] = "viridis"
    heatmap_cmap: str | list[str] = "plasma"

    # Line styles
    line_width: float = 2.0
    marker_size: float = 6.0

    # Font settings
    font_family: str = "sans-serif"
    font_size: float = 10.0
    title_size: float = 14.0
    label_size: float = 12.0

    # Additional rcParams
    rcparams: dict[str, Any] = field(default_factory=dict)

    def get_color_cycle(self) -> list[str]:
        """Get the full color cycle for plotting multiple series."""
        return self.primary_colors + self.secondary_colors + self.accent_colors

    def to_rcparams(self) -> dict[str, Any]:
        """Convert theme to matplotlib rcParams dictionary."""
        params = {
            # Figure
            "figure.facecolor": self.figure_facecolor,
            "figure.edgecolor": self.axes_edgecolor,
            # Axes
            "axes.facecolor": self.axes_facecolor,
            "axes.edgecolor": self.axes_edgecolor,
            "axes.labelcolor": self.label_color,
            "axes.titlecolor": self.title_color,
            "axes.prop_cycle": f"cycler('color', {self.get_color_cycle()})",
            "axes.grid": True,
            "axes.axisbelow": True,
            # Grid
            "grid.color": self.grid_color,
            "grid.alpha": self.grid_alpha,
            "grid.linestyle": self.grid_linestyle,
            # Text
            "text.color": self.text_color,
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            # Ticks
            "xtick.color": self.tick_color,
            "ytick.color": self.tick_color,
            "xtick.labelsize": self.font_size,
            "ytick.labelsize": self.font_size,
            # Lines
            "lines.linewidth": self.line_width,
            "lines.markersize": self.marker_size,
            # Legend
            "legend.framealpha": 0.9,
            "legend.edgecolor": self.axes_edgecolor,
            "legend.facecolor": self.axes_facecolor,
            # Image
            "image.cmap": (
                self.heatmap_cmap if isinstance(self.heatmap_cmap, str) else "viridis"
            ),
        }
        params.update(self.rcparams)
        return params


# =============================================================================
# Theme Definitions
# =============================================================================

# Scientific Violet - The reference theme from the user's screenshots
SCIENTIFIC_VIOLET = PlotTheme(
    name="Scientific Violet",
    description="Purple/magenta data with blue fits and green contours on light background",
    # Light blue-tinted background
    figure_facecolor="#f0f4f8",
    axes_facecolor="#e8f4fc",
    axes_edgecolor="#666666",
    # Text
    text_color="#333333",
    title_color="#1a1a2e",
    label_color="#333333",
    tick_color="#555555",
    # Grid
    grid_color="#b8c9d9",
    grid_alpha=0.6,
    grid_linestyle="-",
    # Primary: Purple/Violet/Magenta for histograms and data points
    primary_colors=["#9333EA", "#A855F7", "#C084FC", "#D8B4FE"],
    primary_color="#9333EA",
    primary_alpha=0.75,
    # Secondary: Blue/Cyan for fitted curves
    secondary_colors=["#3B82F6", "#60A5FA", "#06B6D4", "#22D3EE"],
    secondary_color="#3B82F6",
    # Accent: Green/Teal for highlights
    accent_colors=["#10B981", "#34D399", "#6EE7B7", "#A7F3D0"],
    accent_color="#10B981",
    # Contour colormap: green-yellow-cyan gradient
    contour_cmap="YlGnBu",
    heatmap_cmap="plasma",
    # Style
    line_width=2.5,
    marker_size=8.0,
    font_family="sans-serif",
    font_size=11.0,
    title_size=14.0,
    label_size=12.0,
)

# Scientific Violet Dark - Dark mode version
SCIENTIFIC_VIOLET_DARK = PlotTheme(
    name="Scientific Violet Dark",
    description="Purple/magenta data on dark background",
    figure_facecolor="#1a1a2e",
    axes_facecolor="#16213e",
    axes_edgecolor="#4a5568",
    text_color="#e2e8f0",
    title_color="#f7fafc",
    label_color="#e2e8f0",
    tick_color="#a0aec0",
    grid_color="#2d3748",
    grid_alpha=0.4,
    primary_colors=["#A855F7", "#C084FC", "#D8B4FE", "#E9D5FF"],
    primary_color="#A855F7",
    primary_alpha=0.85,
    secondary_colors=["#60A5FA", "#93C5FD", "#22D3EE", "#67E8F9"],
    secondary_color="#60A5FA",
    accent_colors=["#34D399", "#6EE7B7", "#A7F3D0", "#D1FAE5"],
    accent_color="#34D399",
    contour_cmap="viridis",
    heatmap_cmap="magma",
    line_width=2.5,
    marker_size=8.0,
)

# Catppuccin Mocha - Dark theme matching the app theme
CATPPUCCIN_MOCHA = PlotTheme(
    name="Catppuccin Mocha",
    description="Warm dark theme with pastel accents",
    figure_facecolor="#1e1e2e",
    axes_facecolor="#181825",
    axes_edgecolor="#45475a",
    text_color="#cdd6f4",
    title_color="#cdd6f4",
    label_color="#bac2de",
    tick_color="#a6adc8",
    grid_color="#313244",
    grid_alpha=0.5,
    primary_colors=["#cba6f7", "#f5c2e7", "#f38ba8", "#fab387"],
    primary_color="#cba6f7",
    primary_alpha=0.85,
    secondary_colors=["#89b4fa", "#74c7ec", "#89dceb", "#94e2d5"],
    secondary_color="#89b4fa",
    accent_colors=["#a6e3a1", "#94e2d5", "#f9e2af", "#fab387"],
    accent_color="#a6e3a1",
    contour_cmap="viridis",
    heatmap_cmap="magma",
)

# Catppuccin Latte - Light theme matching the app theme
CATPPUCCIN_LATTE = PlotTheme(
    name="Catppuccin Latte",
    description="Warm light theme with pastel accents",
    figure_facecolor="#eff1f5",
    axes_facecolor="#e6e9ef",
    axes_edgecolor="#9ca0b0",
    text_color="#4c4f69",
    title_color="#4c4f69",
    label_color="#5c5f77",
    tick_color="#6c6f85",
    grid_color="#ccd0da",
    grid_alpha=0.6,
    primary_colors=["#8839ef", "#ea76cb", "#d20f39", "#fe640b"],
    primary_color="#8839ef",
    primary_alpha=0.8,
    secondary_colors=["#1e66f5", "#209fb5", "#04a5e5", "#179299"],
    secondary_color="#1e66f5",
    accent_colors=["#40a02b", "#179299", "#df8e1d", "#fe640b"],
    accent_color="#40a02b",
    contour_cmap="coolwarm",
    heatmap_cmap="plasma",
)

# Dracula - Dark theme
DRACULA = PlotTheme(
    name="Dracula",
    description="Dark theme with vibrant colors",
    figure_facecolor="#282a36",
    axes_facecolor="#21222c",
    axes_edgecolor="#44475a",
    text_color="#f8f8f2",
    title_color="#f8f8f2",
    label_color="#f8f8f2",
    tick_color="#6272a4",
    grid_color="#44475a",
    grid_alpha=0.4,
    primary_colors=["#bd93f9", "#ff79c6", "#ffb86c", "#f1fa8c"],
    primary_color="#bd93f9",
    primary_alpha=0.85,
    secondary_colors=["#8be9fd", "#50fa7b", "#ff5555", "#f1fa8c"],
    secondary_color="#8be9fd",
    accent_colors=["#50fa7b", "#f1fa8c", "#ffb86c", "#ff79c6"],
    accent_color="#50fa7b",
    contour_cmap="viridis",
    heatmap_cmap="inferno",
)

# Nord - Cool dark theme
NORD = PlotTheme(
    name="Nord",
    description="Arctic, north-bluish color palette",
    figure_facecolor="#2e3440",
    axes_facecolor="#3b4252",
    axes_edgecolor="#4c566a",
    text_color="#eceff4",
    title_color="#eceff4",
    label_color="#e5e9f0",
    tick_color="#d8dee9",
    grid_color="#434c5e",
    grid_alpha=0.4,
    primary_colors=["#b48ead", "#bf616a", "#d08770", "#ebcb8b"],
    primary_color="#b48ead",
    primary_alpha=0.85,
    secondary_colors=["#88c0d0", "#81a1c1", "#5e81ac", "#8fbcbb"],
    secondary_color="#88c0d0",
    accent_colors=["#a3be8c", "#8fbcbb", "#ebcb8b", "#d08770"],
    accent_color="#a3be8c",
    contour_cmap="cool",
    heatmap_cmap="plasma",
)

# Solarized Dark
SOLARIZED_DARK = PlotTheme(
    name="Solarized Dark",
    description="Precision colors for machines and people",
    figure_facecolor="#002b36",
    axes_facecolor="#073642",
    axes_edgecolor="#586e75",
    text_color="#839496",
    title_color="#93a1a1",
    label_color="#839496",
    tick_color="#657b83",
    grid_color="#073642",
    grid_alpha=0.5,
    primary_colors=["#d33682", "#6c71c4", "#cb4b16", "#dc322f"],
    primary_color="#d33682",
    primary_alpha=0.85,
    secondary_colors=["#268bd2", "#2aa198", "#859900", "#b58900"],
    secondary_color="#268bd2",
    accent_colors=["#859900", "#2aa198", "#b58900", "#cb4b16"],
    accent_color="#859900",
    contour_cmap="coolwarm",
    heatmap_cmap="magma",
)

# Solarized Light
SOLARIZED_LIGHT = PlotTheme(
    name="Solarized Light",
    description="Light version of Solarized",
    figure_facecolor="#fdf6e3",
    axes_facecolor="#eee8d5",
    axes_edgecolor="#93a1a1",
    text_color="#657b83",
    title_color="#586e75",
    label_color="#657b83",
    tick_color="#839496",
    grid_color="#eee8d5",
    grid_alpha=0.6,
    primary_colors=["#d33682", "#6c71c4", "#cb4b16", "#dc322f"],
    primary_color="#d33682",
    primary_alpha=0.8,
    secondary_colors=["#268bd2", "#2aa198", "#859900", "#b58900"],
    secondary_color="#268bd2",
    accent_colors=["#859900", "#2aa198", "#b58900", "#cb4b16"],
    accent_color="#859900",
    contour_cmap="coolwarm",
    heatmap_cmap="plasma",
)

# One Dark
ONE_DARK = PlotTheme(
    name="One Dark",
    description="Atom One Dark inspired theme",
    figure_facecolor="#282c34",
    axes_facecolor="#21252b",
    axes_edgecolor="#3e4451",
    text_color="#abb2bf",
    title_color="#e6e6e6",
    label_color="#abb2bf",
    tick_color="#5c6370",
    grid_color="#3e4451",
    grid_alpha=0.4,
    primary_colors=["#c678dd", "#e06c75", "#d19a66", "#e5c07b"],
    primary_color="#c678dd",
    primary_alpha=0.85,
    secondary_colors=["#61afef", "#56b6c2", "#98c379", "#e5c07b"],
    secondary_color="#61afef",
    accent_colors=["#98c379", "#56b6c2", "#e5c07b", "#d19a66"],
    accent_color="#98c379",
    contour_cmap="viridis",
    heatmap_cmap="inferno",
)

# Gruvbox Dark
GRUVBOX_DARK = PlotTheme(
    name="Gruvbox Dark",
    description="Retro groove color scheme",
    figure_facecolor="#282828",
    axes_facecolor="#1d2021",
    axes_edgecolor="#504945",
    text_color="#ebdbb2",
    title_color="#fbf1c7",
    label_color="#ebdbb2",
    tick_color="#a89984",
    grid_color="#3c3836",
    grid_alpha=0.4,
    primary_colors=["#d3869b", "#fb4934", "#fe8019", "#fabd2f"],
    primary_color="#d3869b",
    primary_alpha=0.85,
    secondary_colors=["#83a598", "#8ec07c", "#b8bb26", "#fabd2f"],
    secondary_color="#83a598",
    accent_colors=["#b8bb26", "#8ec07c", "#fabd2f", "#fe8019"],
    accent_color="#b8bb26",
    contour_cmap="YlOrRd",
    heatmap_cmap="inferno",
)

# Material Dark
MATERIAL_DARK = PlotTheme(
    name="Material Dark",
    description="Material Design dark theme",
    figure_facecolor="#263238",
    axes_facecolor="#1e272c",
    axes_edgecolor="#455a64",
    text_color="#eceff1",
    title_color="#ffffff",
    label_color="#eceff1",
    tick_color="#90a4ae",
    grid_color="#37474f",
    grid_alpha=0.4,
    primary_colors=["#ce93d8", "#f48fb1", "#ffab91", "#fff59d"],
    primary_color="#ce93d8",
    primary_alpha=0.85,
    secondary_colors=["#81d4fa", "#80cbc4", "#a5d6a7", "#fff59d"],
    secondary_color="#81d4fa",
    accent_colors=["#a5d6a7", "#80cbc4", "#fff59d", "#ffab91"],
    accent_color="#a5d6a7",
    contour_cmap="viridis",
    heatmap_cmap="plasma",
)

# Tokyo Night
TOKYO_NIGHT = PlotTheme(
    name="Tokyo Night",
    description="Clean dark theme inspired by Tokyo city lights",
    figure_facecolor="#1a1b26",
    axes_facecolor="#16161e",
    axes_edgecolor="#3b4261",
    text_color="#a9b1d6",
    title_color="#c0caf5",
    label_color="#a9b1d6",
    tick_color="#565f89",
    grid_color="#24283b",
    grid_alpha=0.4,
    primary_colors=["#bb9af7", "#f7768e", "#ff9e64", "#e0af68"],
    primary_color="#bb9af7",
    primary_alpha=0.85,
    secondary_colors=["#7aa2f7", "#7dcfff", "#9ece6a", "#e0af68"],
    secondary_color="#7aa2f7",
    accent_colors=["#9ece6a", "#7dcfff", "#e0af68", "#ff9e64"],
    accent_color="#9ece6a",
    contour_cmap="viridis",
    heatmap_cmap="magma",
)

# Classic Light - Simple professional light theme
CLASSIC_LIGHT = PlotTheme(
    name="Classic Light",
    description="Clean professional light theme",
    figure_facecolor="#ffffff",
    axes_facecolor="#fafafa",
    axes_edgecolor="#333333",
    text_color="#333333",
    title_color="#111111",
    label_color="#333333",
    tick_color="#666666",
    grid_color="#dddddd",
    grid_alpha=0.6,
    primary_colors=["#7c3aed", "#db2777", "#ea580c", "#ca8a04"],
    primary_color="#7c3aed",
    primary_alpha=0.8,
    secondary_colors=["#2563eb", "#0891b2", "#059669", "#ca8a04"],
    secondary_color="#2563eb",
    accent_colors=["#059669", "#0891b2", "#ca8a04", "#ea580c"],
    accent_color="#059669",
    contour_cmap="RdYlBu_r",
    heatmap_cmap="plasma",
)

# Classic Dark
CLASSIC_DARK = PlotTheme(
    name="Classic Dark",
    description="Clean professional dark theme",
    figure_facecolor="#1a1a1a",
    axes_facecolor="#222222",
    axes_edgecolor="#555555",
    text_color="#e0e0e0",
    title_color="#ffffff",
    label_color="#e0e0e0",
    tick_color="#aaaaaa",
    grid_color="#333333",
    grid_alpha=0.4,
    primary_colors=["#a78bfa", "#f472b6", "#fb923c", "#fbbf24"],
    primary_color="#a78bfa",
    primary_alpha=0.85,
    secondary_colors=["#60a5fa", "#22d3ee", "#34d399", "#fbbf24"],
    secondary_color="#60a5fa",
    accent_colors=["#34d399", "#22d3ee", "#fbbf24", "#fb923c"],
    accent_color="#34d399",
    contour_cmap="viridis",
    heatmap_cmap="inferno",
)

# =============================================================================
# Theme Registry
# =============================================================================

PLOT_THEMES: dict[str, PlotTheme] = {
    "scientific_violet": SCIENTIFIC_VIOLET,
    "scientific_violet_dark": SCIENTIFIC_VIOLET_DARK,
    "catppuccin_mocha": CATPPUCCIN_MOCHA,
    "catppuccin_latte": CATPPUCCIN_LATTE,
    "dracula": DRACULA,
    "nord": NORD,
    "solarized_dark": SOLARIZED_DARK,
    "solarized_light": SOLARIZED_LIGHT,
    "one_dark": ONE_DARK,
    "gruvbox_dark": GRUVBOX_DARK,
    "material_dark": MATERIAL_DARK,
    "tokyo_night": TOKYO_NIGHT,
    "classic_light": CLASSIC_LIGHT,
    "classic_dark": CLASSIC_DARK,
}

# Default theme
DEFAULT_THEME = "scientific_violet"


def get_theme(name: str) -> PlotTheme:
    """Get a theme by name.

    Args:
        name: Theme name (case-insensitive, underscores/hyphens/spaces normalized)

    Returns:
        PlotTheme instance

    Raises:
        KeyError: If theme not found
    """
    # Normalize name
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    if normalized not in PLOT_THEMES:
        available = ", ".join(sorted(PLOT_THEMES.keys()))
        raise KeyError(f"Theme '{name}' not found. Available: {available}")
    return PLOT_THEMES[normalized]


def get_theme_names() -> list[str]:
    """Get list of all available theme names."""
    return sorted(PLOT_THEMES.keys())


def register_theme(name: str, theme: PlotTheme) -> None:
    """Register a custom theme.

    Args:
        name: Theme identifier (will be normalized)
        theme: PlotTheme instance
    """
    assert name is not None, "name must be provided"
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    PLOT_THEMES[normalized] = theme
