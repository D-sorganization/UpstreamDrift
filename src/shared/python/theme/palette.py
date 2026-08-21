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
