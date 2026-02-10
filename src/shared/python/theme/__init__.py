"""Unified theme system for UpstreamDrift.

Source of Truth: UpstreamDrift (this repository)
Consumers:
    - Tools repo: src/shared/theme/ (vendored copy)
    - Gasification_Model: vendor/ud-tools/ (vendored copy)
Cross-repo install: pip install upstream-drift-shared[theme]

This package provides a consistent, sophisticated visual identity across
all components of the application:

- Colors: Unified color palette for UI and data visualization
- Typography: Cross-platform font stacks and sizing
- Matplotlib: Professional plot styling
- ThemeManager: Dynamic theme switching (Light/Dark/High Contrast)

Usage:
    from shared.python.theme import Colors, Sizes, apply_golf_suite_style

    # Apply matplotlib styling globally
    apply_golf_suite_style()

    # Use colors in Qt widgets
    widget.setStyleSheet(f"background-color: {Colors.BG_SURFACE};")

    # Create styled fonts
    from shared.python.theme import get_display_font
    label.setFont(get_display_font(size=16))

    # Dynamic theme switching
    from shared.python.theme import ThemeManager, ThemePreset
    manager = ThemeManager.instance()
    manager.set_theme(ThemePreset.DARK)
"""

from .colors import (
    CHART_COLORS,
    DEFAULT_COLORS,
    ColorPalette,
    Colors,
    get_qcolor,
    get_rgba,
)
from .fleet_adapter import (
    FLEET_THEMES,
    fleet_to_theme_colors,
    get_fleet_theme_names,
    is_fleet_available,
)
from .matplotlib_style import (
    GOLF_SUITE_STYLE,
    apply_golf_suite_style,
    create_styled_figure,
    get_chart_color,
    style_for_export,
)
from .theme_manager import (
    DARK_THEME,
    HIGH_CONTRAST_THEME,
    LIGHT_THEME,
    THEME_PRESETS,
    ThemeColors,
    ThemeManager,
    ThemePreset,
    get_current_colors,
    get_theme_manager,
    set_theme,
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
    # Theme Manager
    "DARK_THEME",
    "HIGH_CONTRAST_THEME",
    "LIGHT_THEME",
    "THEME_PRESETS",
    "ThemeColors",
    "ThemeManager",
    "ThemePreset",
    "get_current_colors",
    "get_theme_manager",
    "set_theme",
    # Fleet-wide Theme Adapter
    "FLEET_THEMES",
    "fleet_to_theme_colors",
    "get_fleet_theme_names",
    "is_fleet_available",
]
