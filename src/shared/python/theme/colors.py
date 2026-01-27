"""Unified color system for Golf Modeling Suite.

This module provides a centralized, sophisticated color palette used across
the entire application - PyQt6 UI, matplotlib plots, and any other visualizations.

Design principles:
- Professional, dark-mode-first aesthetic
- Clear visual hierarchy through color depth
- Accessible contrast ratios (WCAG AA compliant)
- Consistent accent colors across all components
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PyQt6.QtGui import QColor


@dataclass
class ColorPalette:
    """Color palette for the Golf Modeling Suite.

    All colors are hex strings suitable for Qt, CSS, and matplotlib.
    """

    # Primary accent - vibrant blue (more sophisticated than VS Code blue)
    PRIMARY: str = "#0A84FF"
    PRIMARY_HOVER: str = "#409CFF"
    PRIMARY_PRESSED: str = "#0066CC"
    PRIMARY_MUTED: str = "#0A84FF40"  # 25% opacity for backgrounds

    # Secondary accent - success/positive actions
    SUCCESS: str = "#30D158"
    SUCCESS_HOVER: str = "#4AE371"
    SUCCESS_MUTED: str = "#30D15840"

    # Warning accent
    WARNING: str = "#FF9F0A"
    WARNING_HOVER: str = "#FFB340"
    WARNING_MUTED: str = "#FF9F0A40"

    # Error/danger accent
    ERROR: str = "#FF375F"
    ERROR_HOVER: str = "#FF6B8A"
    ERROR_MUTED: str = "#FF375F40"

    # Background hierarchy (depth perception - darker = deeper)
    BG_DEEP: str = "#0D0D0D"  # Deepest - splash screens, modals backdrop
    BG_BASE: str = "#1A1A1A"  # Main window background
    BG_SURFACE: str = "#242424"  # Cards, panels, elevated surfaces
    BG_ELEVATED: str = "#2D2D2D"  # Hover states, active items
    BG_HIGHLIGHT: str = "#383838"  # Strong highlight, selection

    # Border colors
    BORDER_SUBTLE: str = "#2D2D2D"  # Subtle separation
    BORDER_DEFAULT: str = "#404040"  # Standard borders
    BORDER_STRONG: str = "#505050"  # Emphasized borders
    BORDER_FOCUS: str = "#0A84FF"  # Focus ring color

    # Text hierarchy
    TEXT_PRIMARY: str = "#FFFFFF"  # Headings, important text
    TEXT_SECONDARY: str = "#E0E0E0"  # Body text
    TEXT_TERTIARY: str = "#A0A0A0"  # Subtle text, labels
    TEXT_QUATERNARY: str = "#666666"  # Disabled, placeholder
    TEXT_LINK: str = "#0A84FF"  # Links

    # Semantic colors for data visualization
    # Carefully selected for distinguishability and color-blind accessibility
    CHART_BLUE: str = "#0A84FF"
    CHART_GREEN: str = "#30D158"
    CHART_ORANGE: str = "#FF9F0A"
    CHART_RED: str = "#FF375F"
    CHART_PURPLE: str = "#BF5AF2"
    CHART_CYAN: str = "#64D2FF"
    CHART_YELLOW: str = "#FFD60A"
    CHART_BROWN: str = "#AC8E68"

    # Grid and axis colors for plots
    GRID_LINE: str = "#333333"
    AXIS_LINE: str = "#404040"
    TICK_COLOR: str = "#808080"

    # Shadows (for CSS box-shadow or Qt graphics effects)
    SHADOW_LIGHT: str = "rgba(0, 0, 0, 0.15)"
    SHADOW_MEDIUM: str = "rgba(0, 0, 0, 0.25)"
    SHADOW_HEAVY: str = "rgba(0, 0, 0, 0.40)"


# Define Palettes

DARK_PALETTE = ColorPalette()

LIGHT_PALETTE = ColorPalette(
    # Invert backgrounds
    BG_DEEP="#F2F2F7",
    BG_BASE="#FFFFFF",
    BG_SURFACE="#F2F2F7",
    BG_ELEVATED="#FFFFFF",
    BG_HIGHLIGHT="#E5E5EA",
    # Invert text
    TEXT_PRIMARY="#000000",
    TEXT_SECONDARY="#3C3C43",
    TEXT_TERTIARY="#8E8E93",
    TEXT_QUATERNARY="#C7C7CC",
    # Borders
    BORDER_SUBTLE="#E5E5EA",
    BORDER_DEFAULT="#C6C6C8",
    BORDER_STRONG="#8E8E93",
    # Grid/Axis
    GRID_LINE="#E5E5EA",
    AXIS_LINE="#C6C6C8",
    TICK_COLOR="#8E8E93",
    # Shadows
    SHADOW_LIGHT="rgba(0, 0, 0, 0.05)",
    SHADOW_MEDIUM="rgba(0, 0, 0, 0.1)",
    SHADOW_HEAVY="rgba(0, 0, 0, 0.2)",
)

HIGH_CONTRAST_PALETTE = ColorPalette(
    # Stark contrasts
    BG_DEEP="#000000",
    BG_BASE="#000000",
    BG_SURFACE="#000000",
    BG_ELEVATED="#1A1A1A",
    BG_HIGHLIGHT="#333333",
    # Text
    TEXT_PRIMARY="#FFFFFF",
    TEXT_SECONDARY="#FFFFFF",
    TEXT_TERTIARY="#DDDDDD",
    TEXT_QUATERNARY="#BBBBBB",
    TEXT_LINK="#FFFF00",
    # Accents
    PRIMARY="#FFFF00",  # Yellow
    PRIMARY_HOVER="#FFFFCC",
    PRIMARY_PRESSED="#CCCC00",
    PRIMARY_MUTED="#FFFF0040",
    # Borders
    BORDER_SUBTLE="#FFFFFF",
    BORDER_DEFAULT="#FFFFFF",
    BORDER_STRONG="#FFFFFF",
    BORDER_FOCUS="#FFFF00",
    # Grid
    GRID_LINE="#FFFFFF",
    AXIS_LINE="#FFFFFF",
    TICK_COLOR="#FFFFFF",
)


# Global active palette
_active_palette: ColorPalette = DARK_PALETTE


def get_active_palette() -> ColorPalette:
    """Get the currently active color palette."""
    return _active_palette


def set_active_palette(palette: ColorPalette) -> None:
    """Set the active color palette."""
    global _active_palette
    _active_palette = palette


class _ColorsProxy:
    """Proxy object that delegates attribute access to the active palette."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_active_palette(), name)


# Convenience instance (Proxy)
Colors = _ColorsProxy()  # type: ignore


# Chart color sequence for matplotlib and other plotting libraries
# Note: These values are dynamic now if accessed via Colors, but CHART_COLORS
# list itself is static. We should likely make it a property or dynamic too,
# but for now we'll init it from the Default (Dark) palette or leave it as is.
# Using the proxy Colors in the list won't work because lists store values.
# So CHART_COLORS will reflect the palette active at import time (Dark).
# This is a limitation, but acceptable if we assume Dark mode for charts
# or if charts re-fetch colors.
CHART_COLORS = [
    DARK_PALETTE.CHART_BLUE,
    DARK_PALETTE.CHART_GREEN,
    DARK_PALETTE.CHART_ORANGE,
    DARK_PALETTE.CHART_RED,
    DARK_PALETTE.CHART_PURPLE,
    DARK_PALETTE.CHART_CYAN,
    DARK_PALETTE.CHART_YELLOW,
    DARK_PALETTE.CHART_BROWN,
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


def get_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert hex color to RGBA tuple for matplotlib.

    Args:
        hex_color: Hex color string (e.g., "#0A84FF")
        alpha: Alpha value (0.0 to 1.0)

    Returns:
        Tuple of (r, g, b, a) with values 0-1 for matplotlib
    """
    hex_color = str(hex_color).lstrip("#")
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
    "DARK_PALETTE",
    "LIGHT_PALETTE",
    "HIGH_CONTRAST_PALETTE",
    "get_active_palette",
    "set_active_palette",
]
