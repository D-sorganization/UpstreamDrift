"""Theme provider protocol for the portable chat dock widget.

The chat widget previously hard-imported ``theme.theme_manager`` which
required ``theme`` to be on ``sys.path``. That coupling blocked reuse from
applications that vendor the chat package without the ``theme`` package.

Instead, callers now inject any object that implements
:class:`ThemeProviderProtocol`. A :class:`_DefaultDarkTheme` fallback is
provided so the widget remains fully functional with no extra wiring
(matches the dark-mode defaults that ``_get_theme_colors`` previously
returned via its ``dict.get(..., default)`` fallbacks).

Tools issue #2766.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Hardcoded dark-mode defaults — these mirror the inline ``dict.get(...)``
# fallbacks used throughout ``_chat_dock_widget_qt.py`` so existing visuals
# are preserved when no real theme provider is supplied.
_DEFAULT_DARK_COLORS: dict[str, str] = {
    "bg": "#1e1e1e",
    "group_bg": "#2d2d2d",
    "input_bg": "#252526",
    "text": "#e0e0e0",
    "text_secondary": "#888",
    "border": "#444",
    "button_hover": "#ffaa33",
    "accent": "#58a6ff",
}


@runtime_checkable
class ThemeProviderProtocol(Protocol):
    """Minimal interface the chat dock needs from a theme source.

    Compatible with ``theme.theme_manager.ThemeManager`` out of the box —
    inject ``get_theme_manager()`` directly when running inside an app
    that ships the ``theme`` package.
    """

    def get_current_colors(self) -> dict[str, str]:
        """Return the active theme's color map (key -> hex color)."""
        ...


class _DefaultDarkTheme:
    """Fallback theme used when no provider is injected.

    Returns the same dark-mode color values that ``_get_theme_colors``
    previously fell back to via inline ``dict.get(..., "<hex>")`` calls.
    """

    def get_current_colors(self) -> dict[str, str]:
        return dict(_DEFAULT_DARK_COLORS)
