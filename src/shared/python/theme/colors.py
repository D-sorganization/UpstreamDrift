"""Unified color definitions for the fleet-wide theme system.

This module defines all built-in themes and color utilities used across
the D-sorganization repository fleet.

Theme Structure:
    Each theme is a dictionary with the following keys:
    - name: Display name of the theme
    - bg: Main background color
    - group_bg: Group box/card background
    - border: Standard border color
    - text: Primary text color
    - text_secondary: Secondary text color
    - label: Label text color
    - focus: Focus ring/highlight color
    - input_bg: Input field background
    - accent: Primary accent color
    - title_bg: Title/header background
    - title_border: Title/header border
    - table_header: Table header background
    - table_alt: Alternating table row background
    - button_hover: Button hover state background
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtGui import QColor

logger = logging.getLogger(__name__)

# Required base color keys for all themes
THEME_COLOR_KEYS: tuple[str, ...] = (
    "bg",
    "group_bg",
    "border",
    "text",
    "text_secondary",
    "label",
    "focus",
    "input_bg",
    "accent",
    "title_bg",
    "title_border",
    "table_header",
    "table_alt",
    "button_hover",
)

# Optional semantic color keys (added in v2.0)
SEMANTIC_COLOR_KEYS: tuple[str, ...] = (
    "success",
    "warning",
    "error",
    "info",
    "link",
    "link_hover",
    "selection_bg",
    "selection_text",
)


def _load_themes_from_json() -> dict[str, dict[str, str]] | None:
    """Load theme definitions from the canonical themes.json file.

    Returns:
        Dictionary of theme definitions, or None if JSON not available.
    """
    json_path = (
        Path(__file__).parent.parent.parent / "theme-definitions" / "themes.json"
    )
    if not json_path.exists():
        return None

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        themes: dict[str, dict[str, str]] = {}
        for theme_def in data.get("themes", {}).values():
            # Convert kebab-case ID to display name and merge colors + semantic
            flat: dict[str, str] = {"name": theme_def["name"]}
            flat.update(theme_def.get("colors", {}))
            flat.update(theme_def.get("semantic", {}))
            themes[theme_def["name"]] = flat

        if themes:
            logger.debug("Loaded %d themes from %s", len(themes), json_path)
            return themes
    except (PermissionError, OSError) as e:
        logger.warning("Failed to load themes from JSON: %s", e)

    return None


def _load_chart_colors_from_json() -> list[str] | None:
    """Load chart colors from themes.json."""
    json_path = (
        Path(__file__).parent.parent.parent / "theme-definitions" / "themes.json"
    )
    if not json_path.exists():
        return None

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        chart_colors: list[str] | None = data.get("chartColors")
        return chart_colors
    except (PermissionError, OSError):
        return None


# ============================================================================
# BUILT-IN THEMES
# ============================================================================
# Themes are loaded from themes.json when available, with hardcoded fallback.
# When adding a new theme, edit src/shared/theme-definitions/themes.json.

_HARDCODED_BUILTIN_THEMES: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------
    # Standard Light/Dark Themes
    # ------------------------------------------------------------------
    "Light": {
        "name": "Light",
        "bg": "#ffffff",
        "group_bg": "#f8f9fa",
        "border": "#ced4da",
        "text": "#212529",
        "text_secondary": "#495057",
        "label": "#6c757d",
        "focus": "#80bdff",
        "input_bg": "#ffffff",
        "accent": "#5a8fc4",
        "title_bg": "#e3f2fd",
        "title_border": "#90caf9",
        "table_header": "#e9ecef",
        "table_alt": "#f8f9fa",
        "button_hover": "#4a7ba7",
    },
    "Dark": {
        "name": "Dark",
        "bg": "#1a1d23",
        "group_bg": "#24272e",
        "border": "#3a3f4a",
        "text": "#e1e4e8",
        "text_secondary": "#c9d1d9",
        "label": "#8b949e",
        "focus": "#58a6ff",
        "input_bg": "#0d1117",
        "accent": "#4a7ba7",
        "title_bg": "#2d3748",
        "title_border": "#4a7ba7",
        "table_header": "#2d3748",
        "table_alt": "#24272e",
        "button_hover": "#5a8fc4",
    },
    # ------------------------------------------------------------------
    # Neutral/Professional Themes
    # ------------------------------------------------------------------
    "Slate Gray": {
        "name": "Slate Gray",
        "bg": "#f5f5f5",
        "group_bg": "#ebebeb",
        "border": "#c0c0c0",
        "text": "#333333",
        "text_secondary": "#4a4a4a",
        "label": "#666666",
        "focus": "#555555",
        "input_bg": "#ffffff",
        "accent": "#546e7a",
        "title_bg": "#cfd8dc",
        "title_border": "#78909c",
        "table_header": "#e0e0e0",
        "table_alt": "#f5f5f5",
        "button_hover": "#455a64",
    },
    # ------------------------------------------------------------------
    # Nature-Inspired Themes
    # ------------------------------------------------------------------
    "Ocean Blue": {
        "name": "Ocean Blue",
        "bg": "#e8f4f8",
        "group_bg": "#d0e8f2",
        "border": "#90c4d4",
        "text": "#0d3b4f",
        "text_secondary": "#1a5570",
        "label": "#2d6a85",
        "focus": "#3498db",
        "input_bg": "#ffffff",
        "accent": "#2980b9",
        "title_bg": "#b8dce8",
        "title_border": "#5dade2",
        "table_header": "#c8e6f5",
        "table_alt": "#d8ecf5",
        "button_hover": "#1f6a8a",
    },
    "Forest Green": {
        "name": "Forest Green",
        "bg": "#f0f5f0",
        "group_bg": "#e0ebe0",
        "border": "#a8c4a8",
        "text": "#1e4620",
        "text_secondary": "#2d5a2f",
        "label": "#3d6b3f",
        "focus": "#4caf50",
        "input_bg": "#ffffff",
        "accent": "#388e3c",
        "title_bg": "#c8e6c9",
        "title_border": "#66bb6a",
        "table_header": "#d5ead6",
        "table_alt": "#e5f0e5",
        "button_hover": "#2e7d32",
    },
    # ------------------------------------------------------------------
    # Editor/IDE Themes
    # ------------------------------------------------------------------
    "Monokai": {
        "name": "Monokai",
        "bg": "#272822",
        "group_bg": "#3e3d32",
        "border": "#75715e",
        "text": "#f8f8f2",
        "text_secondary": "#ae81ff",
        "label": "#e6db74",
        "focus": "#a6e22e",
        "input_bg": "#171814",
        "accent": "#f92672",
        "title_bg": "#383830",
        "title_border": "#f92672",
        "table_header": "#3e3d32",
        "table_alt": "#272822",
        "button_hover": "#e6db74",
    },
    "Dracula": {
        "name": "Dracula",
        "bg": "#282a36",
        "group_bg": "#343746",
        "border": "#6272a4",
        "text": "#f8f8f2",
        "text_secondary": "#bd93f9",
        "label": "#8be9fd",
        "focus": "#ff79c6",
        "input_bg": "#191a21",
        "accent": "#ff5555",
        "title_bg": "#44475a",
        "title_border": "#bd93f9",
        "table_header": "#44475a",
        "table_alt": "#282a36",
        "button_hover": "#ff79c6",
    },
    "One Dark": {
        "name": "One Dark",
        "bg": "#282c34",
        "group_bg": "#30363f",
        "border": "#5c6370",
        "text": "#abb2bf",
        "text_secondary": "#56b6c2",
        "label": "#e5c07b",
        "focus": "#61afef",
        "input_bg": "#21252b",
        "accent": "#98c379",
        "title_bg": "#353b45",
        "title_border": "#e06c75",
        "table_header": "#353b45",
        "table_alt": "#282c34",
        "button_hover": "#c678dd",
    },
    "Gitpod Dark": {
        "name": "Gitpod Dark",
        "bg": "#0d1117",
        "group_bg": "#161b22",
        "border": "#30363d",
        "text": "#c9d1d9",
        "text_secondary": "#8b949e",
        "label": "#ffb45b",
        "focus": "#12b5cb",
        "input_bg": "#010409",
        "accent": "#12b5cb",
        "title_bg": "#21262d",
        "title_border": "#ffb45b",
        "table_header": "#21262d",
        "table_alt": "#161b22",
        "button_hover": "#0e9dab",
    },
    # ------------------------------------------------------------------
    # Office/Productivity Themes
    # ------------------------------------------------------------------
    "MS Word": {
        "name": "MS Word",
        "bg": "#ffffff",
        "group_bg": "#f3f3f3",
        "border": "#d1d1d1",
        "text": "#000000",
        "text_secondary": "#333333",
        "label": "#666666",
        "focus": "#2b579a",
        "input_bg": "#ffffff",
        "accent": "#2b579a",
        "title_bg": "#deecf9",
        "title_border": "#2b579a",
        "table_header": "#e6e6e6",
        "table_alt": "#f9f9f9",
        "button_hover": "#1e3f6f",
    },
    "MS Excel": {
        "name": "MS Excel",
        "bg": "#ffffff",
        "group_bg": "#f3f3f3",
        "border": "#d1d1d1",
        "text": "#000000",
        "text_secondary": "#333333",
        "label": "#666666",
        "focus": "#217346",
        "input_bg": "#ffffff",
        "accent": "#217346",
        "title_bg": "#e2f0d9",
        "title_border": "#217346",
        "table_header": "#e6e6e6",
        "table_alt": "#f0f7ec",
        "button_hover": "#185c37",
    },
    "Legal Pad": {
        "name": "Legal Pad",
        "bg": "#ffffc0",
        "group_bg": "#fff8a8",
        "border": "#d4c97a",
        "text": "#2d2d00",
        "text_secondary": "#4a4a00",
        "label": "#6b6b00",
        "focus": "#b8860b",
        "input_bg": "#fffff0",
        "accent": "#b8860b",
        "title_bg": "#fff59d",
        "title_border": "#c9a227",
        "table_header": "#f5e6a3",
        "table_alt": "#fffacd",
        "button_hover": "#8b6914",
    },
    # ------------------------------------------------------------------
    # High Contrast / Accessibility
    # ------------------------------------------------------------------
    "High Contrast": {
        "name": "High Contrast",
        "bg": "#000000",
        "group_bg": "#1a1a1a",
        "border": "#ffffff",
        "text": "#ffffff",
        "text_secondary": "#ffffff",
        "label": "#00ffff",
        "focus": "#00ffff",
        "input_bg": "#000000",
        "accent": "#00ffff",
        "title_bg": "#333333",
        "title_border": "#00ffff",
        "table_header": "#333333",
        "table_alt": "#1a1a1a",
        "button_hover": "#00cccc",
    },
}


# ============================================================================
# CHART COLORS
# ============================================================================
# Consistent chart colors for matplotlib and other plotting libraries.
# These are designed for distinguishability and color-blind accessibility.

_HARDCODED_CHART_COLORS: list[str] = [
    "#0A84FF",  # Blue
    "#30D158",  # Green
    "#FF9F0A",  # Orange
    "#FF375F",  # Red
    "#BF5AF2",  # Purple
    "#64D2FF",  # Cyan
    "#FFD60A",  # Yellow
    "#AC8E68",  # Brown
]

# Load from JSON if available, otherwise use hardcoded definitions
BUILTIN_THEMES: dict[str, dict[str, str]] = (
    _load_themes_from_json() or _HARDCODED_BUILTIN_THEMES
)
CHART_COLORS: list[str] = _load_chart_colors_from_json() or _HARDCODED_CHART_COLORS


# ============================================================================
# COLOR UTILITIES
# ============================================================================


def is_valid_hex_color(value: str) -> bool:
    """Return True if value is a valid hexadecimal color string.

    Accepts 3, 4, 6, or 8 hex digit formats (with or without leading #).
    4 and 8 digit formats include an alpha channel.

    Args:
        value: Color string to validate (e.g., "#ff0000", "#f00", "#ff000080")

    Returns:
        True if valid hex color, False otherwise
    """
    value = value.strip()
    if not value:
        return False

    value = value.removeprefix("#")

    return bool(re.fullmatch(r"[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8}", value))


def normalise_hex_color(value: str) -> str:
    """Normalise a hex color into #rrggbb format.

    Args:
        value: Color string (e.g., "#f00", "ff0000", "#FF0000")

    Returns:
        Normalized color string in #rrggbb format

    Raises:
        ValueError: If value is not a valid hex color
    """
    if not is_valid_hex_color(value):
        raise ValueError(f"Invalid colour value: {value!r}")

    stripped = value.strip()
    stripped = stripped.removeprefix("#")

    if len(stripped) == 3:
        stripped = "".join(ch * 2 for ch in stripped)

    return f"#{stripped.lower()}"


def get_qcolor(hex_color: str) -> QColor:
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
        hex_color: Hex color string (e.g., "#0A84FF" or "#0A84FF40")
        alpha: Alpha value (0.0 to 1.0), overrides alpha in hex if present

    Returns:
        Tuple of (r, g, b, a) with values 0-1 for matplotlib
    """
    if hex_color is None:
        raise ValueError("hex_color must be provided")
    hex_color = hex_color.lstrip("#")

    if len(hex_color) == 8:  # Has alpha component
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        a = int(hex_color[6:8], 16) / 255
        return (r, g, b, a * alpha)
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r, g, b, alpha)


def get_matplotlib_colors(theme: dict[str, str]) -> dict[str, str | float]:
    """Get matplotlib-compatible colors from a theme.

    Returns a dictionary with colors suitable for matplotlib figure styling.

    Args:
        theme: Theme dictionary

    Returns:
        Dictionary with matplotlib color settings
    """
    is_dark = _is_dark_theme(theme)

    return {
        "figure.facecolor": theme["bg"],
        "axes.facecolor": theme["group_bg"],
        "axes.edgecolor": theme["border"],
        "axes.labelcolor": theme["text"],
        "text.color": theme["text"],
        "xtick.color": theme["text_secondary"],
        "ytick.color": theme["text_secondary"],
        "grid.color": theme["border"],
        "legend.facecolor": theme["group_bg"],
        "legend.edgecolor": theme["border"],
        # Determine appropriate grid alpha based on theme darkness
        "grid.alpha": 0.3 if is_dark else 0.5,
    }


def _is_dark_theme(theme: dict[str, str]) -> bool:
    """Determine if a theme is dark based on background color.

    Args:
        theme: Theme dictionary

    Returns:
        True if the theme appears to be dark
    """
    bg = theme.get("bg", "#ffffff").lstrip("#")
    if len(bg) >= 6:
        # Calculate relative luminance
        r = int(bg[0:2], 16) / 255
        g = int(bg[2:4], 16) / 255
        b = int(bg[4:6], 16) / 255
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance < 0.5
    return False


def is_dark_theme(theme_name: str) -> bool:
    """Check if a theme name refers to a dark theme.

    Args:
        theme_name: Name of the theme

    Returns:
        True if the theme is dark
    """
    if theme_name in BUILTIN_THEMES:
        return _is_dark_theme(BUILTIN_THEMES[theme_name])
    return False


__all__ = [
    "BUILTIN_THEMES",
    "CHART_COLORS",
    "SEMANTIC_COLOR_KEYS",
    "THEME_COLOR_KEYS",
    "get_matplotlib_colors",
    "get_qcolor",
    "get_rgba",
    "is_dark_theme",
    "is_valid_hex_color",
    "normalise_hex_color",
]
