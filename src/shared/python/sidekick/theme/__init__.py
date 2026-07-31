"""Theme management system for the UpstreamDrift fleet.

This subpackage re-exports the fleet-wide shared theme system,
making it available as ``upstream_drift_tools.theme``.

The canonical source is ``shared.python.theme`` (in this repository).
This module exists so external consumers (e.g., MEB_Conversion) can
``pip install upstream_drift_tools`` and import themes without needing
a git submodule or direct path manipulation.

Usage::

    from sidekick.theme import (
        BUILTIN_THEMES,
        THEME_COLOR_KEYS,
        ThemeManager,
        get_theme_manager,
    )

All symbols from the installed top-level ``theme`` package are re-exported here.
"""

from __future__ import annotations

import logging
import sys

_logger = logging.getLogger(__name__)

# Dynamically import and re-export everything from the canonical theme package.
try:
    if sys.modules.get("theme") is None and "theme" in sys.modules:
        raise ImportError("theme module disabled")

    # Re-export all public symbols
    # Protocol re-exports (no PyQt6 dependency)
    from shared.python.theme import (
        BUILTIN_THEMES,
        THEME_COLOR_KEYS,
        StylesheetGenerator,
        ThemeProvider,
        ThemeSwitcher,
        generate_minimal_stylesheet,
        generate_stylesheet,
        get_matplotlib_colors,
        get_rgba,
        is_dark_theme,
        is_valid_hex_color,
        normalise_hex_color,
    )

    _THEME_AVAILABLE = True

except ImportError as exc:
    _logger.warning("Failed to import theme module: %s", exc)
    _THEME_AVAILABLE = False

# PyQt6-dependent re-exports
if _THEME_AVAILABLE:
    try:
        from shared.python.theme import (
            ColorFieldEditor,
            ColorPickerButton,
            CustomThemeDialog,
            CustomThemeEditor,
            ThemedWindowMixin,
            ThemeListItem,
            ThemeManager,
            ThemeManagerDialog,
            ThemePreviewWidget,
            apply_theme_to_window,
            create_theme_menu,
            get_qcolor,
            get_theme_manager,
            setup_themed_app,
        )

        _PYQT6_AVAILABLE = True
    except ImportError:
        _PYQT6_AVAILABLE = False
        ThemeManager = None
        get_theme_manager = None
else:
    _PYQT6_AVAILABLE = False

__all__ = [
    "BUILTIN_THEMES",
    "ColorFieldEditor",
    "ColorPickerButton",
    "CustomThemeDialog",
    "CustomThemeEditor",
    "StylesheetGenerator",
    "THEME_COLOR_KEYS",
    "ThemeListItem",
    "ThemeManager",
    "ThemeManagerDialog",
    "ThemePreviewWidget",
    "ThemeProvider",
    "ThemeSwitcher",
    "ThemedWindowMixin",
    "apply_theme_to_window",
    "create_theme_menu",
    "generate_minimal_stylesheet",
    "generate_stylesheet",
    "get_matplotlib_colors",
    "get_qcolor",
    "get_rgba",
    "get_theme_manager",
    "is_dark_theme",
    "is_valid_hex_color",
    "normalise_hex_color",
    "setup_themed_app",
]
