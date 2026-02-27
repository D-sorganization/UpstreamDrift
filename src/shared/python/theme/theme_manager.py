"""Theme manager for fleet-wide theme control.

This module provides a centralized ThemeManager for switching themes
across all PyQt6 applications in the D-sorganization fleet.

The ThemeManager supports:
- Singleton pattern for global theme state
- Signal-based notifications for theme changes
- Theme persistence via QSettings
- Custom theme support
- Theme inheritance for embedded applications
"""

from __future__ import annotations

import json
import logging
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar

from PyQt6.QtCore import QObject, QSettings, pyqtSignal

from .colors import BUILTIN_THEMES, THEME_COLOR_KEYS, normalise_hex_color
from .stylesheets import generate_stylesheet

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

# Default settings organization and application name
DEFAULT_SETTINGS_ORG = "D-sorganization"
DEFAULT_SETTINGS_APP = "FleetTheme"


class ThemeManager(QObject):
    """Singleton theme manager for application-wide theme control.

    The ThemeManager provides:
    - Centralized theme management across windows
    - Signal-based theme change notifications
    - Custom theme persistence
    - Theme inheritance for docked/embedded applications

    Usage:
        manager = ThemeManager.instance()
        manager.change_theme("Dark")
        colors = manager.get_current_colors()

    Signals:
        themeChanged(str): Emitted when the theme changes, with the new theme name
    """

    # Singleton instance
    _instance: ClassVar[ThemeManager | None] = None

    # Signal emitted when theme changes
    themeChanged = pyqtSignal(str)

    # Settings key for custom themes
    CUSTOM_THEME_STORAGE_KEY = "custom_themes"

    def __init__(
        self,
        main_window: QWidget | None = None,
        app_context: str | None = None,
        settings_org: str | None = None,
        settings_app: str | None = None,
    ) -> None:
        """Initialize the theme manager.

        Args:
            main_window: Optional main application window to apply themes to.
            app_context: Optional context name for sub-application theming.
                        When set, the application can inherit from global theme.
            settings_org: QSettings organization name. Defaults to "D-sorganization".
            settings_app: QSettings application name. Defaults to "FleetTheme".
        """
        super().__init__()

        self.main_window = main_window
        self.app_context = app_context
        self._settings_org = settings_org or DEFAULT_SETTINGS_ORG
        self._settings_app = settings_app or DEFAULT_SETTINGS_APP

        self.settings = QSettings(self._settings_org, self._settings_app)
        self.custom_themes: dict[str, dict[str, str]] = self._load_custom_themes()
        self._registered_windows: list[weakref.ReferenceType[QWidget]] = []

        # Resolve initial theme
        self.current_theme = self._resolve_effective_theme()

        logger.info(
            "ThemeManager initialized: theme=%s, context=%s",
            self.current_theme,
            self.app_context or "Global",
        )

    @classmethod
    def instance(
        cls,
        main_window: QWidget | None = None,
        app_context: str | None = None,
        settings_org: str | None = None,
        settings_app: str | None = None,
    ) -> ThemeManager:
        """Get the singleton ThemeManager instance.

        On first call, creates the instance with the provided parameters.
        Subsequent calls return the existing instance (parameters are ignored).

        Args:
            main_window: Optional main window (only used on first call)
            app_context: Optional app context (only used on first call)
            settings_org: Optional settings org (only used on first call)
            settings_app: Optional settings app (only used on first call)

        Returns:
            The singleton ThemeManager instance
        """
        if cls._instance is None:
            cls._instance = cls(
                main_window=main_window,
                app_context=app_context,
                settings_org=settings_org,
                settings_app=settings_app,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        This is primarily useful for testing.
        """
        cls._instance = None

    # =========================================================================
    # Theme Resolution
    # =========================================================================

    def _resolve_effective_theme(self) -> str:
        """Resolve the effective theme name, handling inheritance."""
        if self.app_context:
            pref = str(
                self.settings.value(f"theme_{self.app_context}", "Inherit", type=str)
            )
            if pref == "Inherit":
                return str(self.settings.value("theme", "Light", type=str))
            return pref

        return str(self.settings.value("theme", "Light", type=str))

    def get_theme_preference(self) -> str:
        """Get the current theme preference.

        For sub-applications with context, this may return "Inherit".

        Returns:
            Theme preference string
        """
        if self.app_context:
            return str(
                self.settings.value(f"theme_{self.app_context}", "Inherit", type=str)
            )
        return str(self.settings.value("theme", "Light", type=str))

    # =========================================================================
    # Theme Queries
    # =========================================================================

    def get_available_themes(self) -> list[str]:
        """Return all built-in and custom theme names.

        Returns:
            List of theme names
        """
        themes = list(BUILTIN_THEMES.keys()) + sorted(self.custom_themes.keys())
        if self.app_context:
            themes.insert(0, "Inherit")
        return themes

    def get_builtin_themes(self) -> list[str]:
        """Return the built-in theme names.

        Returns:
            List of built-in theme names
        """
        return list(BUILTIN_THEMES.keys())

    def get_custom_theme_names(self) -> list[str]:
        """Return the user-defined theme names.

        Returns:
            List of custom theme names
        """
        return sorted(self.custom_themes.keys())

    def get_current_theme_name(self) -> str:
        """Return the active theme name (resolved).

        Returns:
            Current theme name
        """
        return self.current_theme

    def get_current_colors(self) -> dict[str, str]:
        """Return the color dictionary for the current theme.

        Returns:
            Dictionary of color key -> hex color value
        """
        return dict(self._get_theme_dict(self.current_theme))

    def get_theme_colors(self, theme_name: str) -> dict[str, str] | None:
        """Return a copy of the color mapping for a theme.

        Args:
            theme_name: Name of the theme

        Returns:
            Color dictionary or None if theme doesn't exist
        """
        if theme_name in BUILTIN_THEMES:
            return dict(BUILTIN_THEMES[theme_name])
        if theme_name in self.custom_themes:
            return dict(self.custom_themes[theme_name])
        return None

    # Alias used by custom theme dialogs
    get_theme_definition = get_theme_colors

    def get_theme_stylesheet(self, theme_name: str) -> str:
        """Get the stylesheet for a specific theme without applying it.

        Args:
            theme_name: Name of the theme

        Returns:
            QSS stylesheet string
        """
        if not self._theme_exists(theme_name):
            theme_name = "Light"

        theme = self._get_theme_dict(theme_name)
        return generate_stylesheet(theme)

    def get_current_stylesheet(self) -> str:
        """Get the stylesheet for the current theme.

        Returns:
            QSS stylesheet string
        """
        return self.get_theme_stylesheet(self.current_theme)

    # =========================================================================
    # Theme Application
    # =========================================================================

    def change_theme(self, theme_name: str) -> None:
        """Change to a new theme.

        Args:
            theme_name: Name of the theme to apply, or "Inherit" for sub-apps
        """
        # Handle 'Inherit' case
        if theme_name == "Inherit":
            if not self.app_context:
                logger.warning("Cannot set Global preference to Inherit")
                return

            self.settings.setValue(f"theme_{self.app_context}", "Inherit")
            effective_theme = self.settings.value("theme", "Light", type=str)
            self.current_theme = effective_theme
            self.apply_theme()
            return

        if not self._theme_exists(theme_name):
            logger.warning("Theme '%s' not found, ignoring", theme_name)
            return

        # Save preference
        if self.app_context:
            self.settings.setValue(f"theme_{self.app_context}", theme_name)
        else:
            self.settings.setValue("theme", theme_name)

        self.current_theme = theme_name
        self.apply_theme()
        self.themeChanged.emit(self.current_theme)
        logger.info("Changed theme to: %s", theme_name)

    def apply_theme(self) -> None:
        """Apply the current theme to the main window and registered windows."""
        if not self._theme_exists(self.current_theme):
            logger.warning(
                "Theme '%s' not available, falling back to Light", self.current_theme
            )
            self.current_theme = "Light"

        stylesheet = self.get_current_stylesheet()

        if self.main_window is not None:
            self.main_window.setStyleSheet(stylesheet)

        self._apply_theme_to_registered_windows(stylesheet)
        logger.debug("Applied theme: %s", self.current_theme)

    def apply_theme_to_window(self, window: QWidget) -> None:
        """Apply the current theme to a window and register it for updates.

        Args:
            window: Window to apply theme to
        """
        stylesheet = self.get_current_stylesheet()
        window.setStyleSheet(stylesheet)
        self._register_window(window)
        logger.debug("Applied theme %s to window", self.current_theme)

    def apply_theme_by_name(self, window: QWidget, theme_name: str) -> None:
        """Apply a specific theme to a window without registering it.

        This is useful for preview dialogs or one-off styling.

        Args:
            window: Window to apply theme to
            theme_name: Name of the theme to apply
        """
        if not self._theme_exists(theme_name):
            logger.warning("Theme '%s' not found, ignoring", theme_name)
            return

        stylesheet = self.get_theme_stylesheet(theme_name)
        window.setStyleSheet(stylesheet)
        logger.debug("Applied specific theme '%s' to window", theme_name)

    # =========================================================================
    # Custom Theme Management
    # =========================================================================

    def save_custom_theme(
        self,
        theme_name: str,
        colors: dict[str, str],
        apply_immediately: bool = False,
    ) -> str:
        """Save a user-defined theme.

        Args:
            theme_name: Name for the custom theme
            colors: Dictionary of color key -> hex color value
            apply_immediately: If True, switch to the new theme

        Returns:
            The saved theme name

        Raises:
            ValueError: If theme name is empty or conflicts with built-in
        """
        cleaned_name = theme_name.strip()
        if not cleaned_name:
            raise ValueError("Theme name cannot be empty.")

        if cleaned_name in BUILTIN_THEMES:
            raise ValueError(
                f"Theme name '{cleaned_name}' conflicts with a built-in theme."
            )

        normalised_colors = self._validate_custom_theme_colors(colors)
        self.custom_themes[cleaned_name] = normalised_colors
        self._persist_custom_themes()

        if apply_immediately:
            self.change_theme(cleaned_name)

        logger.info("Saved custom theme: %s", cleaned_name)
        return cleaned_name

    def save_current_theme_as_custom(self, theme_name: str) -> str:
        """Save the current theme as a custom theme with a new name.

        Args:
            theme_name: Name for the new custom theme

        Returns:
            The saved theme name

        Raises:
            ValueError: If the name is invalid
        """
        current_colors = self.get_current_colors()
        colors_only = {k: v for k, v in current_colors.items() if k in THEME_COLOR_KEYS}
        return self.save_custom_theme(theme_name, colors_only, apply_immediately=False)

    def delete_custom_theme(self, theme_name: str) -> bool:
        """Remove a stored custom theme.

        Args:
            theme_name: Name of the theme to delete

        Returns:
            True if deleted, False if not found
        """
        if theme_name not in self.custom_themes:
            return False

        del self.custom_themes[theme_name]
        self._persist_custom_themes()

        if self.current_theme == theme_name:
            self.change_theme("Light")

        logger.info("Deleted custom theme: %s", theme_name)
        return True

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _theme_exists(self, theme_name: str) -> bool:
        """Check if a theme exists."""
        return theme_name in BUILTIN_THEMES or theme_name in self.custom_themes

    def _get_theme_dict(self, theme_name: str) -> dict[str, str]:
        """Get the color dictionary for a theme."""
        if theme_name in BUILTIN_THEMES:
            return BUILTIN_THEMES[theme_name]
        return self.custom_themes.get(theme_name, BUILTIN_THEMES["Light"])

    def _load_custom_themes(self) -> dict[str, dict[str, str]]:
        """Load custom themes from settings."""
        raw_value = self.settings.value(self.CUSTOM_THEME_STORAGE_KEY, "{}", type=str)
        try:
            data = json.loads(raw_value)
            if not isinstance(data, dict):
                raise ValueError("Invalid custom theme data structure")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to load custom themes: %s", exc)
            data = {}

        cleaned: dict[str, dict[str, str]] = {}
        for name, colors in data.items():
            if not isinstance(name, str) or not isinstance(colors, dict):
                continue
            try:
                filtered = {k: v for k, v in colors.items() if k in THEME_COLOR_KEYS}
                theme_def = self._validate_custom_theme_colors(filtered)
                cleaned[name] = theme_def
            except ValueError:
                logger.debug("Discarded invalid custom theme '%s'", name)

        return cleaned

    def _persist_custom_themes(self) -> None:
        """Save custom themes to settings."""
        to_save = {
            name: {k: v for k, v in colors.items() if k in THEME_COLOR_KEYS}
            for name, colors in self.custom_themes.items()
        }
        self.settings.setValue(
            self.CUSTOM_THEME_STORAGE_KEY, json.dumps(to_save, indent=2)
        )
        self.settings.sync()

    def _validate_custom_theme_colors(
        self, colors: dict[str, str] | Iterable[tuple[str, str]]
    ) -> dict[str, str]:
        """Validate and normalize custom theme colors."""
        items: Iterable[tuple[str, str]]
        items = colors.items() if isinstance(colors, dict) else colors

        normalised: dict[str, str] = {}
        for key, value in items:
            if key not in THEME_COLOR_KEYS:
                continue
            normalised[key] = normalise_hex_color(str(value))

        missing_keys = [key for key in THEME_COLOR_KEYS if key not in normalised]
        if missing_keys:
            raise ValueError("Missing colour values for: " + ", ".join(missing_keys))

        return normalised

    def _register_window(self, window: QWidget) -> None:
        """Register a window for theme updates."""
        # Clean up dead references
        for ref in list(self._registered_windows):
            obj = ref()
            if obj is None:
                self._registered_windows.remove(ref)
            elif obj is window:
                return

        self._registered_windows.append(weakref.ref(window))

    def _apply_theme_to_registered_windows(self, stylesheet: str) -> None:
        """Apply stylesheet to all registered windows."""
        alive_refs: list[weakref.ReferenceType[QWidget]] = []
        for ref in self._registered_windows:
            window = ref()
            if window is None:
                continue
            window.setStyleSheet(stylesheet)
            alive_refs.append(ref)

        self._registered_windows = alive_refs


# ============================================================================
# Convenience Functions
# ============================================================================


def get_theme_manager(
    main_window: QWidget | None = None,
    app_context: str | None = None,
    settings_org: str | None = None,
    settings_app: str | None = None,
) -> ThemeManager:
    """Get the singleton ThemeManager instance.

    This is the recommended way to access the theme manager.

    Args:
        main_window: Optional main window (only used on first call)
        app_context: Optional app context (only used on first call)
        settings_org: Optional settings org (only used on first call)
        settings_app: Optional settings app (only used on first call)

    Returns:
        The singleton ThemeManager instance
    """
    return ThemeManager.instance(
        main_window=main_window,
        app_context=app_context,
        settings_org=settings_org,
        settings_app=settings_app,
    )


__all__ = [
    "ThemeManager",
    "get_theme_manager",
]
