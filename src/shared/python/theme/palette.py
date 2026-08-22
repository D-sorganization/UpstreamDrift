"""Theme palette mapping with semantic attribute aliases (issue #8972).

The fleet theme system stores colors as a flat ``dict[str, str]`` keyed by
``THEME_COLOR_KEYS`` (``bg``, ``border``, ``accent``, ...), while several
launcher call sites address colors through richer semantic names
(``bg_elevated``, ``border_default``, ``text_primary``, ...).

:class:`ThemePalette` bridges the two without forking palette data: it is a
plain ``dict`` of the canonical keys that additionally resolves a documented
set of semantic aliases via attribute access.
"""

from __future__ import annotations

#: Semantic attribute name -> canonical ``THEME_COLOR_KEYS`` entry.
#: Every alias maps onto existing palette data; no colors are invented.
SEMANTIC_ALIASES: dict[str, str] = {
    "bg_base": "bg",
    "bg_elevated": "group_bg",
    "bg_highlight": "button_hover",
    "surface_primary": "bg",
    "surface_secondary": "group_bg",
    "surface_tertiary": "title_bg",
    "border_default": "border",
    "border_light": "border",
    "border_strong": "focus",
    "primary": "accent",
    "text_primary": "text",
    "text_tertiary": "label",
    "text_quaternary": "label",
    "text_muted": "label",
}


class ThemePalette(dict):
    """Color mapping supporting both dict access and semantic attributes.

    Invariant: attribute access never invents color values — it resolves
    either a canonical key (``palette.bg``) or a documented alias from
    ``SEMANTIC_ALIASES`` (``palette.bg_elevated`` -> ``palette["group_bg"]``).
    Unknown names raise ``AttributeError`` so ``getattr(palette, name,
    default)`` call sites keep their explicit fallbacks.
    """

    def __getattr__(self, name: str) -> str:
        try:
            return self[name]
        except KeyError:
            pass
        alias = SEMANTIC_ALIASES.get(name)
        if alias is not None and alias in self:
            return self[alias]
        raise AttributeError(
            f"{type(self).__name__} has no color {name!r} "
            f"(canonical keys: {sorted(self)})"
        )


__all__ = ["SEMANTIC_ALIASES", "ThemePalette"]


def _builtin_dark_palette() -> ThemePalette:
    """Build the fallback palette from the built-in "Dark" theme."""
    from src.shared.python.theme.colors import BUILTIN_THEMES

    return ThemePalette(BUILTIN_THEMES["Dark"])


#: Documented fallback palette: the built-in "Dark" theme's color mapping.
#: Derived from ``BUILTIN_THEMES`` — never fork palette data.  Used by
#: launcher call sites as a last resort when no theme manager is available.
DARK_THEME: ThemePalette = _builtin_dark_palette()


def get_current_colors() -> ThemePalette:
    """Return the active theme's color mapping.

    Canonical UD-owned accessor (issue #8972) delegating to the singleton
    ``ThemeManager`` when PyQt6 is available, falling back to the built-in
    Dark palette otherwise (e.g. headless environments).

    Lives here rather than in ``theme/__init__`` because that module is a
    Tools-owned child copy which UpstreamDrift must not edit directly
    (tests/unit/repo_hygiene/test_tools_child_copy_contract.py).

    Postcondition:
        The returned mapping contains every key in ``THEME_COLOR_KEYS``
        (``bg``, ``border``, ``accent``, ...) mapped to a hex color, and
        additionally resolves the semantic attribute aliases documented in
        :data:`SEMANTIC_ALIASES` (``bg_elevated``, ``text_primary``, ...).
    """
    from src.shared.python.theme.colors import THEME_COLOR_KEYS

    try:
        from src.shared.python.theme import get_theme_manager

        colors = ThemePalette(get_theme_manager().get_current_colors())
    except (ImportError, AttributeError, TypeError, RuntimeError):
        # No PyQt6 / no manager singleton available (headless, early startup).
        colors = ThemePalette(DARK_THEME)
    missing = [key for key in THEME_COLOR_KEYS if key not in colors]
    if missing:  # DbC postcondition: complete mapping for launcher consumers
        for key in missing:
            colors[key] = DARK_THEME[key]
    return colors
