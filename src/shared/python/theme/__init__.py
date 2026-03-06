"""Fleet-wide shared theme management system.

This module provides a unified color theme system for all PyQt6 GUI applications
across the D-sorganization repository fleet.

Features:
- 12+ built-in themes (Light, Dark, Monokai, Dracula, One Dark, etc.)
- Custom theme support with persistence
- Theme inheritance for docked applications
- Qt stylesheet generation
- Matplotlib integration for consistent plotting colors
- Signal-based theme change notifications

Usage:
    from shared.python.theme import ThemeManager, get_theme_manager

    # Get singleton instance
    manager = get_theme_manager()

    # Get available themes
    themes = manager.get_available_themes()

    # Change theme
    manager.change_theme("Dark")

    # Apply to a window
    manager.apply_theme_to_window(my_window)

    # Connect to theme changes
    manager.themeChanged.connect(self.on_theme_changed)

    # Access current colors for custom styling
    colors = manager.get_current_colors()
    bg_color = colors["bg"]
"""

from types import SimpleNamespace as _NS

from .colors import (
    BUILTIN_THEMES,
    CHART_COLORS,
    SEMANTIC_COLOR_KEYS,
    THEME_COLOR_KEYS,
    get_matplotlib_colors,
    get_rgba,
    is_dark_theme,
    is_valid_hex_color,
    normalise_hex_color,
)
from .protocols import StylesheetGenerator, ThemeProvider, ThemeSwitcher
from .stylesheets import generate_minimal_stylesheet, generate_stylesheet

# Convenience fallback: a ThemeColors-compatible dark theme.
# Used by launcher code as a safe default when PyQt6 ThemeManager is unavailable.
# Built via fleet_to_theme_colors to get all proper semantic attributes.
try:
    from .fleet_adapter import fleet_to_theme_colors as _f2tc

    DARK_THEME = _f2tc("Dark")
except Exception:
    # Ultimate fallback: minimal SimpleNamespace if fleet adapter is unavailable
    _dark = BUILTIN_THEMES.get("Dark", {})
    DARK_THEME = _NS(  # type: ignore[assignment]
        bg=_dark.get("bg", "#1a1d23"),
        bg_elevated=_dark.get("table_header", "#2a2d35"),
        bg_highlight=_dark.get("title_bg", "#2e3340"),
        border_default=_dark.get("border", "#3a3d45"),
        border_strong=_dark.get("border", "#4a4d55"),
        primary=_dark.get("accent", "#0A84FF"),
        success=_dark.get("success", "#30D158"),
        success_hover="#38e066",
        error=_dark.get("error", "#FF375F"),
        text_primary=_dark.get("text", "#FFFFFF"),
        text_quaternary="#666666",
    )

# PyQt6-dependent imports - only available when PyQt6 is installed
try:
    from .colors import get_qcolor
    from .dialogs import (
        ColorFieldEditor,
        ColorPickerButton,
        CustomThemeDialog,
        CustomThemeEditor,
        ThemeListItem,
        ThemeManagerDialog,
        ThemePreviewWidget,
    )
    from .integration import (
        ThemedWindowMixin,
        apply_theme_to_window,
        create_theme_menu,
        setup_themed_app,
    )
    from .theme_manager import ThemeManager, get_theme_manager

    _PYQT6_AVAILABLE = True
except ImportError:
    _PYQT6_AVAILABLE = False
    ThemeManager = None  # type: ignore[assignment, misc]
    get_theme_manager = None  # type: ignore[assignment]
    get_qcolor = None  # type: ignore[assignment]
    ThemedWindowMixin = None  # type: ignore[assignment, misc]
    apply_theme_to_window = None  # type: ignore[assignment]
    create_theme_menu = None  # type: ignore[assignment]
    setup_themed_app = None  # type: ignore[assignment]
    ColorFieldEditor = None  # type: ignore[assignment, misc]
    ColorPickerButton = None  # type: ignore[assignment, misc]
    CustomThemeDialog = None  # type: ignore[assignment, misc]
    CustomThemeEditor = None  # type: ignore[assignment, misc]
    ThemeListItem = None  # type: ignore[assignment, misc]
    ThemeManagerDialog = None  # type: ignore[assignment, misc]
    ThemePreviewWidget = None  # type: ignore[assignment, misc]

__all__ = [
    # Protocols (no PyQt6 dependency)
    "StylesheetGenerator",
    "ThemeProvider",
    "ThemeSwitcher",
    # Theme manager (requires PyQt6)
    "ThemeManager",
    "get_theme_manager",
    # Integration helpers (requires PyQt6)
    "ThemedWindowMixin",
    "apply_theme_to_window",
    "create_theme_menu",
    "setup_themed_app",
    # Dialogs (requires PyQt6)
    "ColorFieldEditor",
    "ColorPickerButton",
    "CustomThemeDialog",
    "CustomThemeEditor",
    "ThemeListItem",
    "ThemeManagerDialog",
    "ThemePreviewWidget",
    # Color utilities
    "BUILTIN_THEMES",
    "CHART_COLORS",
    "DARK_THEME",
    "SEMANTIC_COLOR_KEYS",
    "THEME_COLOR_KEYS",
    "get_matplotlib_colors",
    "get_qcolor",
    "get_rgba",
    "is_dark_theme",
    "is_valid_hex_color",
    "normalise_hex_color",
    # Stylesheet generation
    "generate_minimal_stylesheet",
    "generate_stylesheet",
]
