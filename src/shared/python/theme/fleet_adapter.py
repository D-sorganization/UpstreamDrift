"""Fleet-wide theme adapter for UpstreamDrift.

This module provides an adapter layer between the fleet-wide shared theme system
(vendor/ud-tools/src/shared/python/theme/) and UpstreamDrift's native ThemeColors
dataclass format.

This enables access to all 14 fleet-wide themes (loaded from themes.json) while
maintaining backward compatibility with UpstreamDrift's existing theme API.
Semantic colors (success, warning, error, info, link) are read from the JSON
source of truth rather than being hardcoded.

Usage:
    from shared.python.theme.fleet_adapter import (
        FLEET_THEMES,
        get_fleet_theme_names,
        fleet_to_theme_colors,
    )

    # Get available fleet themes
    names = get_fleet_theme_names()  # ['Light', 'Dark', 'Monokai', ...]

    # Convert a fleet theme to ThemeColors
    colors = fleet_to_theme_colors("Dracula")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .theme_manager import ThemeColors

logger = logging.getLogger(__name__)

# Load fleet-wide themes from vendor/ud-tools using importlib to avoid
# circular import: this module is imported during src/shared/python/theme/__init__
# initialization, so `from shared.python.theme import ...` would resolve to the
# partially-loaded UpstreamDrift module instead of the vendor module.
_vendor_theme_path = (
    Path(__file__).parent.parent.parent.parent.parent
    / "vendor"
    / "ud-tools"
    / "src"
    / "shared"
    / "python"
    / "theme"
)

# Try to import fleet-wide themes
try:
    import importlib.util

    _colors_path = _vendor_theme_path / "colors.py"
    if not _colors_path.exists():
        raise ImportError(f"vendor theme module not found at {_colors_path}")

    _spec = importlib.util.spec_from_file_location(
        "_vendor_theme_colors", str(_colors_path)
    )
    _vendor_colors = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_vendor_colors)  # type: ignore[union-attr]

    FLEET_THEMES: dict = _vendor_colors.BUILTIN_THEMES
    is_valid_hex_color = _vendor_colors.is_valid_hex_color

    _FLEET_AVAILABLE = True
    logger.info("Fleet-wide theme system available from vendor/ud-tools")

except ImportError as e:
    _FLEET_AVAILABLE = False
    FLEET_THEMES = {}
    logger.warning(
        "Fleet-wide theme module not available (%s). "
        "Run 'git submodule update --init --recursive' to enable.",
        e,
    )

    def is_valid_hex_color(value: str) -> bool:
        """Fallback hex color validator."""
        import re

        value = value.strip()
        if not value:
            return False
        value = value.removeprefix("#")
        if len(value) == 3:
            return bool(re.fullmatch(r"[0-9a-fA-F]{3}", value))
        return bool(re.fullmatch(r"[0-9a-fA-F]{6}", value))


def is_fleet_available() -> bool:
    """Check if the fleet-wide theme system is available."""
    return _FLEET_AVAILABLE


def get_fleet_theme_names() -> list[str]:
    """Get list of available fleet-wide theme names."""
    return list(FLEET_THEMES.keys())


def _is_dark_theme(theme_dict: dict[str, str]) -> bool:
    """Determine if a theme is dark based on background color."""
    bg = theme_dict.get("bg", "#ffffff")
    if not bg.startswith("#"):
        return False

    # Parse hex and check luminance
    hex_val = bg.lstrip("#")
    if len(hex_val) == 3:
        hex_val = "".join(c * 2 for c in hex_val)

    try:
        r = int(hex_val[0:2], 16)
        g = int(hex_val[2:4], 16)
        b = int(hex_val[4:6], 16)
        # Simple luminance check (perceived brightness)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return luminance < 0.5
    except (ValueError, IndexError):
        return False


def _adjust_color_brightness(hex_color: str, factor: float) -> str:
    """Adjust color brightness by a factor (>1 = lighter, <1 = darker)."""
    hex_val = hex_color.lstrip("#")
    if len(hex_val) == 3:
        hex_val = "".join(c * 2 for c in hex_val)

    try:
        r = int(hex_val[0:2], 16)
        g = int(hex_val[2:4], 16)
        b = int(hex_val[4:6], 16)

        r = min(255, max(0, int(r * factor)))
        g = min(255, max(0, int(g * factor)))
        b = min(255, max(0, int(b * factor)))

        return f"#{r:02x}{g:02x}{b:02x}"
    except (ValueError, IndexError):
        return hex_color


def _hex_with_alpha(hex_color: str, alpha: int) -> str:
    """Add alpha channel to hex color (for muted variants)."""
    hex_val = hex_color.lstrip("#")
    if len(hex_val) == 3:
        hex_val = "".join(c * 2 for c in hex_val)
    return f"#{hex_val}{alpha:02x}"


def _extract_base_colors(ft: dict[str, str], is_dark: bool) -> dict[str, str]:
    accent = ft.get("accent", "#0A84FF")
    return {
        "accent": accent,
        "text": ft.get("text", "#FFFFFF" if is_dark else "#1A1A1A"),
        "text_secondary": ft.get("text_secondary", "#E0E0E0" if is_dark else "#404040"),
        "bg": ft.get("bg", "#1A1A1A" if is_dark else "#FFFFFF"),
        "group_bg": ft.get("group_bg", "#242424" if is_dark else "#F8F9FA"),
        "input_bg": ft.get("input_bg", "#0D1117" if is_dark else "#FFFFFF"),
        "border": ft.get("border", "#404040" if is_dark else "#D0D0D0"),
        "focus": ft.get("focus", accent),
        "label": ft.get("label", "#A0A0A0" if is_dark else "#666666"),
    }


def _extract_semantic_colors(ft: dict[str, str], is_dark: bool) -> dict[str, str]:
    return {
        "success": ft.get("success", "#30D158" if is_dark else "#28A745"),
        "warning": ft.get("warning", "#FF9F0A" if is_dark else "#E67E00"),
        "error": ft.get("error", "#FF375F" if is_dark else "#DC3545"),
        "info": ft.get("info", "#64D2FF" if is_dark else "#17A2B8"),
    }


def _build_theme_colors_kwargs(
    theme_name: str,
    ft: dict[str, str],
    is_dark: bool,
    base: dict[str, str],
    semantic: dict[str, str],
) -> dict:
    accent = base["accent"]
    group_bg = base["group_bg"]
    border = base["border"]
    label = base["label"]
    success = semantic["success"]
    warning = semantic["warning"]
    error = semantic["error"]
    info = semantic["info"]

    return {
        "name": theme_name,
        "is_dark": is_dark,
        "primary": accent,
        "primary_hover": _adjust_color_brightness(accent, 1.2),
        "primary_pressed": _adjust_color_brightness(accent, 0.8),
        "primary_muted": _hex_with_alpha(accent, 0x40),
        "success": success,
        "success_hover": _adjust_color_brightness(success, 1.2),
        "success_muted": _hex_with_alpha(success, 0x40),
        "warning": warning,
        "warning_hover": _adjust_color_brightness(warning, 1.2),
        "warning_muted": _hex_with_alpha(warning, 0x40),
        "error": error,
        "error_hover": _adjust_color_brightness(error, 1.2),
        "error_muted": _hex_with_alpha(error, 0x40),
        "bg_deep": base["input_bg"],
        "bg_base": base["bg"],
        "bg_surface": group_bg,
        "bg_elevated": ft.get("table_header", _adjust_color_brightness(group_bg, 1.1)),
        "bg_highlight": ft.get("title_bg", _adjust_color_brightness(group_bg, 1.2)),
        "border_subtle": _adjust_color_brightness(border, 0.8 if is_dark else 1.1),
        "border_default": border,
        "border_strong": _adjust_color_brightness(border, 1.2 if is_dark else 0.9),
        "border_focus": base["focus"],
        "text_primary": base["text"],
        "text_secondary": base["text_secondary"],
        "text_tertiary": label,
        "text_quaternary": _adjust_color_brightness(label, 0.7),
        "text_link": ft.get("link", accent),
        "chart_blue": accent,
        "chart_green": success,
        "chart_orange": warning,
        "chart_red": error,
        "chart_purple": "#BF5AF2" if is_dark else "#9B59B6",
        "chart_cyan": info,
        "chart_yellow": "#FFD60A" if is_dark else "#F0AD4E",
        "chart_brown": "#AC8E68" if is_dark else "#8B6914",
        "grid_line": _adjust_color_brightness(border, 0.7 if is_dark else 1.2),
        "axis_line": border,
        "tick_color": label,
        "shadow_light": "rgba(0, 0, 0, 0.15)" if is_dark else "rgba(0, 0, 0, 0.08)",
        "shadow_medium": "rgba(0, 0, 0, 0.25)" if is_dark else "rgba(0, 0, 0, 0.12)",
        "shadow_heavy": "rgba(0, 0, 0, 0.40)" if is_dark else "rgba(0, 0, 0, 0.20)",
    }


def fleet_to_theme_colors(theme_name: str) -> ThemeColors:
    """Convert a fleet-wide theme to UpstreamDrift's ThemeColors format.

    This maps the simpler fleet theme structure (bg, text, accent, etc.) to
    UpstreamDrift's richer semantic color system (primary, success, warning,
    bg_deep, bg_base, bg_surface, etc.).

    Args:
        theme_name: Name of the fleet theme (e.g., "Light", "Dark", "Dracula")

    Returns:
        ThemeColors dataclass instance

    Raises:
        KeyError: If theme_name is not found in fleet themes
    """
    from .theme_manager import ThemeColors

    if theme_name not in FLEET_THEMES:
        raise KeyError(f"Fleet theme '{theme_name}' not found")

    ft = FLEET_THEMES[theme_name]
    is_dark = _is_dark_theme(ft)
    base = _extract_base_colors(ft, is_dark)
    semantic = _extract_semantic_colors(ft, is_dark)
    kwargs = _build_theme_colors_kwargs(theme_name, ft, is_dark, base, semantic)

    return ThemeColors(**kwargs)


__all__ = [
    "FLEET_THEMES",
    "fleet_to_theme_colors",
    "get_fleet_theme_names",
    "is_fleet_available",
]
