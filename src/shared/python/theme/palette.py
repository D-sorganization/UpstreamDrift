"""Theme palette mapping with semantic attribute aliases (issue #8972, #9037).

The fleet theme system stores colors as a flat ``dict[str, str]`` keyed by
``THEME_COLOR_KEYS`` (``bg``, ``border``, ``accent``, ...), while several
launcher call sites address colors through richer semantic names
(``bg_elevated``, ``border_default``, ``text_primary``, ...).

:class:`ThemePalette` bridges the two without forking palette data: it is a
plain ``dict`` of the canonical keys that additionally resolves a documented
set of semantic aliases via attribute access.

:class:`Colors` provides a typed token container whose attribute access dynamically
delegates to the active theme palette via metaclass interception, ensuring theme
switches immediately reflect in token reads while preserving IDE autocompletion
and type annotations.
"""

from __future__ import annotations

from typing import Any

#: Semantic attribute name -> canonical ``THEME_COLOR_KEYS`` entry.
#: Every alias maps onto existing palette data; no colors are invented.
SEMANTIC_ALIASES: dict[str, str] = {
    "bg_base": "bg",
    "bg_surface": "group_bg",
    "bg_elevated": "group_bg",
    "bg_highlight": "button_hover",
    "bg_deep": "input_bg",
    "surface_primary": "bg",
    "surface_secondary": "group_bg",
    "surface_tertiary": "title_bg",
    "border_default": "border",
    "border_light": "border",
    "border_strong": "focus",
    "border_subtle": "border",
    "primary": "accent",
    "primary_hover": "button_hover",
    "text_primary": "text",
    "text_secondary": "text_secondary",
    "text_tertiary": "label",
    "text_quaternary": "label",
    "text_muted": "label",
    "grid_line": "border",
    "tick_color": "label",
}


class ThemePalette(dict[str, str]):
    """Color mapping supporting both dict access and semantic attributes.

    Invariant: attribute access never invents color values — it resolves
    either a canonical key (``palette.bg``) or a documented alias from
    ``SEMANTIC_ALIASES`` (``palette.bg_elevated`` -> ``palette["group_bg"]``).
    Unknown names raise ``AttributeError`` so ``getattr(palette, name,
    default)`` call sites keep their explicit fallbacks.
    """

    def __getattr__(self, name: str) -> str:
        # 1. Direct key match
        try:
            return str(self[name])
        except KeyError:
            pass
        # 2. Exact alias match
        alias = SEMANTIC_ALIASES.get(name)
        if alias is not None and alias in self:
            return str(self[alias])
        # 3. Lowercase key match
        lower_name = name.lower()
        try:
            return str(self[lower_name])
        except KeyError:
            pass
        # 4. Lowercase alias match
        alias = SEMANTIC_ALIASES.get(lower_name)
        if alias is not None and alias in self:
            return str(self[alias])
        raise AttributeError(
            f"{type(self).__name__} has no color {name!r} "
            f"(canonical keys: {sorted(self)})"
        )

    @classmethod
    def get_current_colors(cls) -> ThemePalette:
        """Return the active theme palette."""
        return get_current_colors()


def _builtin_dark_palette() -> ThemePalette:
    """Build the fallback palette from the built-in "Dark" theme."""
    from .colors import BUILTIN_THEMES

    return ThemePalette(BUILTIN_THEMES["Dark"])


#: Documented fallback palette: the built-in "Dark" theme's color mapping.
#: Derived from ``BUILTIN_THEMES`` — never fork palette data.  Used by
#: launcher call sites as a last resort when no theme manager is available.
DARK_THEME: ThemePalette = _builtin_dark_palette()


def get_current_colors() -> ThemePalette:
    """Return the active theme's color mapping.

    Canonical accessor delegating to the singleton ``ThemeManager`` when
    PyQt6 is available, falling back to the built-in Dark palette otherwise
    (e.g. headless environments).
    """
    from .colors import THEME_COLOR_KEYS

    try:
        from .theme_manager import get_theme_manager

        manager = get_theme_manager()
        if manager is not None and hasattr(manager, "get_current_colors"):
            raw_colors = manager.get_current_colors()
            if raw_colors:
                colors = ThemePalette(raw_colors)
            else:
                colors = ThemePalette(DARK_THEME)
        else:
            colors = ThemePalette(DARK_THEME)
    except (ImportError, AttributeError, TypeError, RuntimeError):
        # No PyQt6 / no manager singleton available (headless, early startup).
        colors = ThemePalette(DARK_THEME)
    missing = [key for key in THEME_COLOR_KEYS if key not in colors]
    if missing:  # DbC postcondition: complete mapping for launcher consumers
        for key in missing:
            colors[key] = DARK_THEME.get(key, "#000000")
    return colors


def _resolve_color_token(target: Any, name: str) -> Any:
    """Helper resolving color token attributes on metaclass or instance."""
    is_cls = type(target) is type or issubclass(type(target), type)
    if (name.startswith("__") and name.endswith("__")) or name == "get_current_colors":
        return (
            type.__getattribute__(target, name)
            if is_cls
            else object.__getattribute__(target, name)
        )
    try:
        return getattr(get_current_colors(), name)
    except AttributeError:
        pass
    return (
        type.__getattribute__(target, name)
        if is_cls
        else object.__getattribute__(target, name)
    )


class _ColorsMeta(type):
    """Metaclass intercepting attribute access to dynamically resolve theme tokens."""

    def __getattribute__(cls, name: str) -> Any:
        return _resolve_color_token(cls, name)


class Colors(metaclass=_ColorsMeta):
    """Dynamic theme color palette accessor.

    Accessing attributes on :class:`Colors` always delegates to the active
    theme returned by :func:`get_current_colors`, ensuring theme switches
    immediately reflect in all token reads while maintaining typed annotations.
    """

    BG_BASE: str = "#1a1d23"
    """Base background color."""

    BG_SURFACE: str = "#24272e"
    """Surface/card background color."""

    BG_ELEVATED: str = "#24272e"
    """Elevated container background color."""

    BG_DEEP: str = "#0d1117"
    """Deep inset/input background color."""

    BG_HIGHLIGHT: str = "#5a8fc4"
    """Highlight background color."""

    BORDER_DEFAULT: str = "#3a3f4a"
    """Default border color."""

    BORDER_SUBTLE: str = "#3a3f4a"
    """Subtle border color."""

    BORDER_STRONG: str = "#58a6ff"
    """Strong/focused border color."""

    BORDER_LIGHT: str = "#3a3f4a"
    """Light border color."""

    PRIMARY: str = "#4a7ba7"
    """Primary action/accent color."""

    PRIMARY_HOVER: str = "#5a8fc4"
    """Primary hover color."""

    ACCENT: str = "#4a7ba7"
    """Theme accent color."""

    TEXT_PRIMARY: str = "#e1e4e8"
    """Primary text color."""

    TEXT_SECONDARY: str = "#c9d1d9"
    """Secondary text color."""

    TEXT_TERTIARY: str = "#8b949e"
    """Tertiary/muted label color."""

    TEXT_QUATERNARY: str = "#8b949e"
    """Quaternary label color."""

    TEXT_MUTED: str = "#8b949e"
    """Muted text color."""

    GRID_LINE: str = "#3a3f4a"
    """Grid line color."""

    TICK_COLOR: str = "#8b949e"
    """Axis tick color."""

    FOCUS: str = "#58a6ff"
    """Focus highlight color."""

    SUCCESS: str = "#30d158"
    """Success status color."""

    WARNING: str = "#ffd60a"
    """Warning status color."""

    ERROR: str = "#ff375f"
    """Error status color."""

    INFO: str = "#64b5f6"
    """Info status color."""

    def __getattribute__(self, name: str) -> Any:
        return _resolve_color_token(self, name)

    @classmethod
    def get_current_colors(cls) -> ThemePalette:
        """Return the active theme palette."""
        return get_current_colors()


__all__ = [
    "DARK_THEME",
    "SEMANTIC_ALIASES",
    "Colors",
    "ThemePalette",
    "get_current_colors",
]
