"""Pure-Python theme helpers for ChatQuickBar.

Split from ``quick_bar`` so the protocol and color-resolution logic can be
imported and tested without triggering the PyQt6 / QtWebSockets DLL load.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ThemeProviderProtocol(Protocol):
    """Minimal protocol for a theme provider.

    Satisfies both the local fallback and any future ``_theme_protocol``
    implementation so callers remain forward-compatible with PR #2766.
    """

    def get_colors(self) -> dict[str, str]:
        """Return a mapping of token names to hex color strings."""
        ...


class _FallbackThemeProvider:
    """Built-in dark-mode defaults used when no theme provider is supplied."""

    _COLORS: dict[str, str] = {
        "background_primary": "#252526",
        "background_secondary": "#2d2d2d",
        "border": "#3c3c3c",
        "text_primary": "#e0e0e0",
        "text_secondary": "#888888",
        "accent": "#FF8800",
        "accent_hover": "#ffaa33",
        "focus": "#4ec9b0",
        "input_background": "#2d2d2d",
        "button_background": "#3c3c3c",
        "button_hover": "#4c4c4c",
        "disabled_background": "#555555",
        "disabled_foreground": "#888888",
        "muted": "#666666",
    }

    def get_colors(self) -> dict[str, str]:
        return dict(self._COLORS)


def _build_system_theme_provider() -> ThemeProviderProtocol:
    """Try to construct a provider backed by the installed theme manager.

    Falls back to :class:`_FallbackThemeProvider` if the theme system is
    unavailable (headless CI, optional dependency not installed, etc.).
    """

    class _SystemThemeProvider:
        """Wraps ``get_theme_manager()`` as a ``ThemeProviderProtocol``."""

        def get_colors(self) -> dict[str, str]:
            try:
                from theme.theme_manager import (  # type: ignore[import-not-found]
                    get_theme_manager,
                )

                mgr = get_theme_manager()
                return mgr.get_current_colors()
            except Exception:  # noqa: BLE001
                logger.debug("Theme manager unavailable, using fallback colors")
                return {}

    return _SystemThemeProvider()


def _resolve_colors(provider: ThemeProviderProtocol) -> dict[str, str]:
    """Map raw theme-manager tokens to quick-bar canonical names.

    Each canonical key keeps a hardcoded hex fallback so the bar is always
    renderable even when the provider returns an incomplete palette.
    """
    raw = provider.get_colors()

    def tok(*keys: str, default: str) -> str:
        for k in keys:
            v = raw.get(k)
            if v:
                return v
        return default

    return {
        "bg": tok("background_primary", "bg", default="#252526"),
        "group_bg": tok("background_secondary", "group_bg", default="#2d2d2d"),
        "border": tok("border", default="#3c3c3c"),
        "text": tok("text_primary", "text", default="#e0e0e0"),
        "text_secondary": tok("text_secondary", default="#888888"),
        "accent": tok("accent", default="#FF8800"),
        "accent_hover": tok("accent_hover", "button_hover", default="#ffaa33"),
        "focus": tok("focus", default="#4ec9b0"),
        "input_bg": tok(
            "input_background", "input_bg", "background_secondary", default="#2d2d2d"
        ),
        "button_bg": tok(
            "button_background", "button_bg", "background_secondary", default="#3c3c3c"
        ),
        "button_hover": tok("button_hover", default="#4c4c4c"),
        "disabled_bg": tok("disabled_background", "disabled_bg", default="#555555"),
        "disabled_fg": tok("disabled_foreground", "disabled_fg", default="#888888"),
        "muted": tok("muted", "label", default="#666666"),
    }
