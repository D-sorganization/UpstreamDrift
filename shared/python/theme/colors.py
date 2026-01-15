"""Unified color system for Golf Modeling Suite.

This module provides a centralized, sophisticated color palette used across
the entire application - PyQt6 UI, matplotlib plots, and any other visualizations.

Design principles:
- Professional, dark-mode-first aesthetic
- Clear visual hierarchy through color depth
- Accessible contrast ratios (WCAG AA compliant)
- Consistent accent colors across all components
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ColorPalette:
    """Immutable color palette for the Golf Modeling Suite.

    All colors are hex strings suitable for Qt, CSS, and matplotlib.
    """

    # Primary accent - vibrant blue (more sophisticated than VS Code blue)
    PRIMARY: ClassVar[str] = "#0A84FF"
    PRIMARY_HOVER: ClassVar[str] = "#409CFF"
    PRIMARY_PRESSED: ClassVar[str] = "#0066CC"
    PRIMARY_MUTED: ClassVar[str] = "#0A84FF40"  # 25% opacity for backgrounds

    # Secondary accent - success/positive actions
    SUCCESS: ClassVar[str] = "#30D158"
    SUCCESS_HOVER: ClassVar[str] = "#4AE371"
    SUCCESS_MUTED: ClassVar[str] = "#30D15840"

    # Warning accent
    WARNING: ClassVar[str] = "#FF9F0A"
    WARNING_HOVER: ClassVar[str] = "#FFB340"
    WARNING_MUTED: ClassVar[str] = "#FF9F0A40"

    # Error/danger accent
    ERROR: ClassVar[str] = "#FF375F"
    ERROR_HOVER: ClassVar[str] = "#FF6B8A"
    ERROR_MUTED: ClassVar[str] = "#FF375F40"

    # Background hierarchy (depth perception - darker = deeper)
    BG_DEEP: ClassVar[str] = "#0D0D0D"  # Deepest - splash screens, modals backdrop
    BG_BASE: ClassVar[str] = "#1A1A1A"  # Main window background
    BG_SURFACE: ClassVar[str] = "#242424"  # Cards, panels, elevated surfaces
    BG_ELEVATED: ClassVar[str] = "#2D2D2D"  # Hover states, active items
    BG_HIGHLIGHT: ClassVar[str] = "#383838"  # Strong highlight, selection

    # Border colors
    BORDER_SUBTLE: ClassVar[str] = "#2D2D2D"  # Subtle separation
    BORDER_DEFAULT: ClassVar[str] = "#404040"  # Standard borders
    BORDER_STRONG: ClassVar[str] = "#505050"  # Emphasized borders
    BORDER_FOCUS: ClassVar[str] = "#0A84FF"  # Focus ring color

    # Text hierarchy
    TEXT_PRIMARY: ClassVar[str] = "#FFFFFF"  # Headings, important text
    TEXT_SECONDARY: ClassVar[str] = "#E0E0E0"  # Body text
    TEXT_TERTIARY: ClassVar[str] = "#A0A0A0"  # Subtle text, labels
    TEXT_QUATERNARY: ClassVar[str] = "#666666"  # Disabled, placeholder
    TEXT_LINK: ClassVar[str] = "#0A84FF"  # Links

    # Semantic colors for data visualization
    # Carefully selected for distinguishability and color-blind accessibility
    CHART_BLUE: ClassVar[str] = "#0A84FF"
    CHART_GREEN: ClassVar[str] = "#30D158"
    CHART_ORANGE: ClassVar[str] = "#FF9F0A"
    CHART_RED: ClassVar[str] = "#FF375F"
    CHART_PURPLE: ClassVar[str] = "#BF5AF2"
    CHART_CYAN: ClassVar[str] = "#64D2FF"
    CHART_YELLOW: ClassVar[str] = "#FFD60A"
    CHART_BROWN: ClassVar[str] = "#AC8E68"

    # Grid and axis colors for plots
    GRID_LINE: ClassVar[str] = "#333333"
    AXIS_LINE: ClassVar[str] = "#404040"
    TICK_COLOR: ClassVar[str] = "#808080"

    # Shadows (for CSS box-shadow or Qt graphics effects)
    SHADOW_LIGHT: ClassVar[str] = "rgba(0, 0, 0, 0.15)"
    SHADOW_MEDIUM: ClassVar[str] = "rgba(0, 0, 0, 0.25)"
    SHADOW_HEAVY: ClassVar[str] = "rgba(0, 0, 0, 0.40)"


# Convenience instance
Colors = ColorPalette()


# Chart color sequence for matplotlib and other plotting libraries
CHART_COLORS = [
    Colors.CHART_BLUE,
    Colors.CHART_GREEN,
    Colors.CHART_ORANGE,
    Colors.CHART_RED,
    Colors.CHART_PURPLE,
    Colors.CHART_CYAN,
    Colors.CHART_YELLOW,
    Colors.CHART_BROWN,
]


# Legacy compatibility mapping for existing code
DEFAULT_COLORS = {
    "primary": Colors.CHART_BLUE,
    "secondary": Colors.CHART_ORANGE,
    "tertiary": Colors.CHART_GREEN,
    "quaternary": Colors.CHART_RED,
    "quinary": Colors.CHART_PURPLE,
    "senary": Colors.CHART_BROWN,
    "accent": Colors.CHART_CYAN,
    "dark": Colors.TEXT_TERTIARY,
    "grid": Colors.GRID_LINE,
}


def get_qcolor(hex_color: str) -> "QColor":
    """Convert hex color string to QColor.

    Args:
        hex_color: Hex color string (e.g., "#0A84FF")

    Returns:
        QColor instance

    Note:
        This is a lazy import to avoid Qt dependency in non-GUI contexts.
    """
    from PyQt6.QtGui import QColor

    return QColor(hex_color)


def get_rgba(hex_color: str, alpha: float = 1.0) -> tuple[int, int, int, float]:
    """Convert hex color to RGBA tuple for matplotlib.

    Args:
        hex_color: Hex color string (e.g., "#0A84FF")
        alpha: Alpha value (0.0 to 1.0)

    Returns:
        Tuple of (r, g, b, a) with values 0-1 for matplotlib
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 8:  # Has alpha component
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        a = int(hex_color[6:8], 16) / 255
        return (r, g, b, a * alpha)
    else:
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return (r, g, b, alpha)


__all__ = [
    "CHART_COLORS",
    "ColorPalette",
    "Colors",
    "DEFAULT_COLORS",
    "get_qcolor",
    "get_rgba",
]
