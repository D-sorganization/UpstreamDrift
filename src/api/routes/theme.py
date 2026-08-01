"""Theme management endpoints for UpstreamDrift.

Provides access to the fleet-wide ThemeManager so the React UI can synchronize
its styling with the PyQt6 desktop launcher. Runtime API images do not install
the desktop PyQt6 stack, so route discovery falls back to a read-only built-in
theme provider when the desktop manager is unavailable.
"""

import logging

from typing import Any

from src.shared.python.theme.api import ThemeColors, create_theme_router
from src.shared.python.theme.colors import BUILTIN_THEMES

logger = logging.getLogger(__name__)


class _StaticThemeManager:
    """PyQt-free theme provider for API runtimes without desktop persistence."""

    def __init__(self, initial_theme: str = "Dark") -> None:
        if initial_theme not in BUILTIN_THEMES:
            raise ValueError(f"unknown built-in theme {initial_theme!r}")
        self._current_theme = initial_theme

    def get_builtin_themes(self) -> list[str]:
        return list(BUILTIN_THEMES)

    def get_custom_theme_names(self) -> list[str]:
        return []

    def get_available_themes(self) -> list[str]:
        return self.get_builtin_themes()

    def get_theme_colors(self, name: str) -> dict[str, str]:
        if name not in BUILTIN_THEMES:
            return {}
        return self._resolved_colors(name)

    def get_current_theme_name(self) -> str:
        return self._current_theme

    def get_current_colors(self) -> dict[str, str]:
        return self._resolved_colors(self._current_theme)

    def change_theme(self, name: str) -> None:
        if name not in BUILTIN_THEMES:
            raise ValueError(f"unknown built-in theme {name!r}")
        self._current_theme = name

    def save_custom_theme(
        self,
        name: str,
        colors: dict[str, str],
        apply: bool = False,
    ) -> str:
        raise ValueError("Custom theme persistence requires the desktop theme manager")

    def delete_custom_theme(self, theme_id: str) -> bool:
        return False

    @staticmethod
    def _resolved_colors(name: str) -> dict[str, str]:
        palette: dict[str, Any] = dict(BUILTIN_THEMES[name])
        resolved: dict[str, Any] = ThemeColors(**palette).as_dict()
        return {
            key: value
            for key, value in resolved.items()
            if key not in {"name", "is_dark"} and isinstance(value, str)
        }


def _build_theme_manager() -> Any:
    try:
        from src.shared.python.theme.theme_manager import ThemeManager

        return ThemeManager.instance(
            settings_org="D-sorganization",
            settings_app="FleetTheme",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Desktop theme manager unavailable; using API-safe built-in themes: %s",
            exc,
        )
        return _StaticThemeManager()


theme_manager = _build_theme_manager()

# Create the router using the shared factory
# This will expose /themes/active, /themes/, /themes/builtin, /themes/custom endpoints
# The /themes prefix ensures proper routing under /api/v1/themes when mounted
router = create_theme_router(theme_manager)
