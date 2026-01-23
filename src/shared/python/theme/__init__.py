"""Unified theme system for Golf Modeling Suite.

This package provides a consistent, sophisticated visual identity across
all components of the application:

- Colors: Unified color palette for UI and data visualization
- Typography: Cross-platform font stacks and sizing
- Matplotlib: Professional plot styling

Usage:
    from shared.python.theme import Colors, Sizes, apply_golf_suite_style

    # Apply matplotlib styling globally
    apply_golf_suite_style()

    # Use colors in Qt widgets
    widget.setStyleSheet(f"background-color: {Colors.BG_SURFACE};")

    # Create styled fonts
    from shared.python.theme import get_display_font
    label.setFont(get_display_font(size=16))
"""

from .colors import (
    CHART_COLORS,
    DEFAULT_COLORS,
    ColorPalette,
    Colors,
    get_qcolor,
    get_rgba,
)
from .matplotlib_style import (
    GOLF_SUITE_STYLE,
    apply_golf_suite_style,
    create_styled_figure,
    get_chart_color,
    style_for_export,
)
from .typography import (
    CSS_FONT_DISPLAY,
    CSS_FONT_MONO,
    CSS_FONT_UI,
    FONT_STACK_DISPLAY,
    FONT_STACK_MONO,
    FONT_STACK_UI,
    FontSizes,
    FontWeights,
    Sizes,
    Weights,
    get_display_font,
    get_mono_font,
    get_qfont,
)

__all__ = [
    # Colors
    "CHART_COLORS",
    "ColorPalette",
    "Colors",
    "DEFAULT_COLORS",
    "get_qcolor",
    "get_rgba",
    # Typography
    "CSS_FONT_DISPLAY",
    "CSS_FONT_MONO",
    "CSS_FONT_UI",
    "FONT_STACK_DISPLAY",
    "FONT_STACK_MONO",
    "FONT_STACK_UI",
    "FontSizes",
    "FontWeights",
    "Sizes",
    "Weights",
    "get_display_font",
    "get_mono_font",
    "get_qfont",
    # Matplotlib
    "GOLF_SUITE_STYLE",
    "apply_golf_suite_style",
    "create_styled_figure",
    "get_chart_color",
    "style_for_export",
]
