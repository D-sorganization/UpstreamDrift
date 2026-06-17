# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Bridge to the fleet shared theme (``shared.python.theme``).

Movement-Optimizer prefers the fleet shared theme (from ``ud-tools``) so it
matches the rest of the fleet and offers the same theme menu. The shared theme
is used automatically wherever ``ud-tools`` is installed (all fleet dev machines
and runners). When it is not importable -- e.g. a bare CI virtualenv -- this
module falls back to bundled fleet colours so the app and tests still run.

The rest of the app imports theme helpers via ``rendering`` (which re-exports
these names), keeping the conditional dependency in exactly one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from shared.python.theme import BUILTIN_THEMES as _SHARED_THEMES
    from shared.python.theme import ThemedWindowMixin as _SharedThemedWindowMixin
    from shared.python.theme import get_theme_manager as _shared_get_theme_manager
    from shared.python.theme.matplotlib_style import apply_plot_theme as _shared_apply_plot_theme
    from shared.python.theme.matplotlib_style import get_chart_color as _shared_get_chart_color

    SHARED_THEME_AVAILABLE = True
    BUILTIN_THEMES: Mapping[str, Mapping[str, str]] = _SHARED_THEMES
    ThemedWindowMixin: type = _SharedThemedWindowMixin
    get_theme_manager = _shared_get_theme_manager
    apply_plot_theme = _shared_apply_plot_theme
    get_chart_color = _shared_get_chart_color

except ImportError:
    SHARED_THEME_AVAILABLE = False

    # Bundled fleet colours (subset of the shared theme's Dark/Light tokens).
    BUILTIN_THEMES = {  # type: ignore[no-redef]
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
        "Light": {
            "name": "Light",
            "bg": "#f5f6f8",
            "group_bg": "#ffffff",
            "border": "#d0d7de",
            "text": "#1f2328",
            "text_secondary": "#3a3f45",
            "label": "#57606a",
            "focus": "#0969da",
            "input_bg": "#ffffff",
            "accent": "#4a7ba7",
            "title_bg": "#eaeef2",
            "title_border": "#4a7ba7",
            "table_header": "#eaeef2",
            "table_alt": "#f5f6f8",
            "button_hover": "#5a8fc4",
        },
    }

    _CHART_COLORS = (
        "#0A84FF",
        "#FF9F0A",
        "#30D158",
        "#FF453A",
        "#BF5AF2",
        "#64D2FF",
        "#FFD60A",
        "#FF6482",
    )

    def get_chart_color(index: int) -> str:  # type: ignore[misc]
        """Return a colour from the bundled accessible chart cycle."""
        return _CHART_COLORS[index % len(_CHART_COLORS)]

    def apply_plot_theme(fig: Any, colors: Mapping[str, str]) -> None:  # type: ignore[misc]
        """Apply bundled theme colours to a matplotlib figure and its axes."""
        bg = colors.get("bg", "#1a1d23")
        panel = colors.get("group_bg", bg)
        fg = colors.get("text_secondary", "#c9d1d9")
        fig.set_facecolor(bg)
        for ax in fig.get_axes():
            ax.set_facecolor(panel)
            ax.tick_params(colors=fg, which="both")
            for spine in ax.spines.values():
                spine.set_color(fg)

    class _FallbackThemeManager:
        """Minimal stand-in for the shared ThemeManager (no Qt signals)."""

        def __init__(self) -> None:
            self._theme = "Dark"

        def get_current_colors(self) -> Mapping[str, str]:
            return BUILTIN_THEMES[self._theme]

        def get_current_theme_name(self) -> str:
            return self._theme

        def get_available_themes(self) -> list[str]:
            return list(BUILTIN_THEMES)

        def change_theme(self, name: str) -> None:
            if name in BUILTIN_THEMES:
                self._theme = name

    _FALLBACK_MANAGER = _FallbackThemeManager()

    def get_theme_manager(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[misc]
        """Return the bundled fallback theme manager."""
        return _FALLBACK_MANAGER

    class ThemedWindowMixin:  # type: ignore[no-redef]
        """No-op theme mixin used when the shared theme is unavailable."""

        def setup_theme_support(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def change_theme(self, _name: str) -> None:
            return None

        def get_current_theme(self) -> str:
            return "Dark"
