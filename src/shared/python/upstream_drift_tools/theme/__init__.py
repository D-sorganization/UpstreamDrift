"""Theme management system for the UpstreamDrift fleet.

This subpackage re-exports the fleet-wide shared theme system,
making it available as ``upstream_drift_tools.theme``.

The canonical source is ``shared.python.theme`` (in this repository).
This module exists so external consumers (e.g., MEB_Conversion) can
``pip install upstream_drift_tools`` and import themes without needing
a git submodule or direct path manipulation.

Usage::

    from upstream_drift_tools.theme import (
        BUILTIN_THEMES,
        THEME_COLOR_KEYS,
        ThemeManager,
        get_theme_manager,
    )

All symbols from ``shared.python.theme`` are re-exported here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure the shared.python.theme sibling package is importable.
# When installed via pip (editable or not), setuptools already
# includes the search paths, but we guard against edge cases.
_shared_python_dir = Path(__file__).resolve().parent.parent.parent
if str(_shared_python_dir) not in sys.path:
    sys.path.insert(0, str(_shared_python_dir))

# Dynamically import and re-export everything from shared.python.theme
try:
    # Re-export all public symbols
    # Protocol re-exports (no PyQt6 dependency)
    from theme import (
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
    logger.warning("Failed to import theme module: %s", exc)
    _THEME_AVAILABLE = False

# PyQt6-dependent re-exports
if _THEME_AVAILABLE:
    try:
        from theme import (
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
    # Core data (no PyQt6 needed)
    "BUILTIN_THEMES",
    "THEME_COLOR_KEYS",
    # Color utilities
    "get_matplotlib_colors",
    "get_rgba",
    "is_dark_theme",
    "is_valid_hex_color",
    "normalise_hex_color",
    # Stylesheet generation
    "generate_minimal_stylesheet",
    "generate_stylesheet",
    # Protocols
    "StylesheetGenerator",
    "ThemeProvider",
    "ThemeSwitcher",
    # PyQt6-dependent (may be None)
    "ThemeManager",
    "get_theme_manager",
    "ThemedWindowMixin",
    "apply_theme_to_window",
    "create_theme_menu",
    "setup_themed_app",
    "get_qcolor",
    # Dialogs
    "ColorFieldEditor",
    "ColorPickerButton",
    "CustomThemeDialog",
    "CustomThemeEditor",
    "ThemeListItem",
    "ThemeManagerDialog",
    "ThemePreviewWidget",
]
