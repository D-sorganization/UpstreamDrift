# ruff: noqa: E501
"""Theme-color helpers shared by chat-dock widgets.

Extracted from ``_chat_dock_widget_qt`` to keep that module under the
1500-line repo budget. Theme provider lookup is intentionally tolerant —
a misbehaving provider must never crash the chat dock (Tools issue #2766).
"""

from __future__ import annotations

from .._theme_protocol import ThemeProviderProtocol, _DefaultDarkTheme


def get_theme_colors(
    theme_provider: ThemeProviderProtocol | None = None,
) -> dict[str, str]:
    """Return the current theme color map from the injected provider.

    Falls back to :class:`_DefaultDarkTheme` so the widget never depends on
    ``theme.theme_manager`` being importable.
    """
    provider: ThemeProviderProtocol = theme_provider or _DefaultDarkTheme()
    try:
        colors: dict[str, str] = provider.get_current_colors()
        return colors
    except Exception:  # noqa: BLE001 - defensive: a misbehaving provider must not crash the widget
        colors = _DefaultDarkTheme().get_current_colors()
        return colors
