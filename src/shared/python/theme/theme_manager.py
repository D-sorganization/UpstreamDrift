"""Theme manager for dynamic theme switching.

This module provides a centralized ThemeManager for switching between
Light, Dark, High Contrast, and fleet-wide themes throughout the application.
Also supports custom user-defined themes with persistence via QSettings.

Usage:
    from shared.python.theme import ThemeManager, ThemePreset

    # Get singleton instance
    manager = ThemeManager.instance()

    # Switch theme
    manager.set_theme(ThemePreset.DARK)

    # Get current colors
    colors = manager.colors

    # Connect to theme changes (PyQt6)
    manager.theme_changed.connect(self.on_theme_changed)
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PyQt6.QtCore import pyqtSignal


class ThemePreset(Enum):
    """Available theme presets."""

    LIGHT = auto()
    DARK = auto()
    HIGH_CONTRAST = auto()


@dataclass(frozen=True)
class ThemeColors:
    """Color definitions for a theme.

    All colors are hex strings suitable for Qt, CSS, and matplotlib.
    """

    # Theme metadata
    name: str
    is_dark: bool

    # Primary accent
    primary: str
    primary_hover: str
    primary_pressed: str
    primary_muted: str

    # Semantic colors
    success: str
    success_hover: str
    success_muted: str
    warning: str
    warning_hover: str
    warning_muted: str
    error: str
    error_hover: str
    error_muted: str

    # Background hierarchy
    bg_deep: str
    bg_base: str
    bg_surface: str
    bg_elevated: str
    bg_highlight: str

    # Border colors
    border_subtle: str
    border_default: str
    border_strong: str
    border_focus: str

    # Text hierarchy
    text_primary: str
    text_secondary: str
    text_tertiary: str
    text_quaternary: str
    text_link: str

    # Chart colors
    chart_blue: str
    chart_green: str
    chart_orange: str
    chart_red: str
    chart_purple: str
    chart_cyan: str
    chart_yellow: str
    chart_brown: str

    # Grid and axis
    grid_line: str
    axis_line: str
    tick_color: str

    # Shadows
    shadow_light: str
    shadow_medium: str
    shadow_heavy: str


# Dark theme (default, current design)
DARK_THEME = ThemeColors(
    name="Dark",
    is_dark=True,
    # Primary accent - vibrant blue
    primary="#0A84FF",
    primary_hover="#409CFF",
    primary_pressed="#0066CC",
    primary_muted="#0A84FF40",
    # Semantic colors
    success="#30D158",
    success_hover="#4AE371",
    success_muted="#30D15840",
    warning="#FF9F0A",
    warning_hover="#FFB340",
    warning_muted="#FF9F0A40",
    error="#FF375F",
    error_hover="#FF6B8A",
    error_muted="#FF375F40",
    # Background hierarchy
    bg_deep="#0D0D0D",
    bg_base="#1A1A1A",
    bg_surface="#242424",
    bg_elevated="#2D2D2D",
    bg_highlight="#383838",
    # Border colors
    border_subtle="#2D2D2D",
    border_default="#404040",
    border_strong="#505050",
    border_focus="#0A84FF",
    # Text hierarchy
    text_primary="#FFFFFF",
    text_secondary="#E0E0E0",
    text_tertiary="#A0A0A0",
    text_quaternary="#666666",
    text_link="#0A84FF",
    # Chart colors
    chart_blue="#0A84FF",
    chart_green="#30D158",
    chart_orange="#FF9F0A",
    chart_red="#FF375F",
    chart_purple="#BF5AF2",
    chart_cyan="#64D2FF",
    chart_yellow="#FFD60A",
    chart_brown="#AC8E68",
    # Grid and axis
    grid_line="#333333",
    axis_line="#404040",
    tick_color="#808080",
    # Shadows
    shadow_light="rgba(0, 0, 0, 0.15)",
    shadow_medium="rgba(0, 0, 0, 0.25)",
    shadow_heavy="rgba(0, 0, 0, 0.40)",
)


# Light theme
LIGHT_THEME = ThemeColors(
    name="Light",
    is_dark=False,
    # Primary accent - slightly darker blue for light bg
    primary="#0066CC",
    primary_hover="#0077E6",
    primary_pressed="#004C99",
    primary_muted="#0066CC20",
    # Semantic colors (slightly darker for light mode)
    success="#28A745",
    success_hover="#34C759",
    success_muted="#28A74520",
    warning="#E67E00",
    warning_hover="#FF9F0A",
    warning_muted="#E67E0020",
    error="#DC3545",
    error_hover="#FF375F",
    error_muted="#DC354520",
    # Background hierarchy (inverted - lighter = deeper)
    bg_deep="#FFFFFF",
    bg_base="#F8F9FA",
    bg_surface="#FFFFFF",
    bg_elevated="#F0F0F0",
    bg_highlight="#E8E8E8",
    # Border colors
    border_subtle="#E8E8E8",
    border_default="#D0D0D0",
    border_strong="#B0B0B0",
    border_focus="#0066CC",
    # Text hierarchy (dark text on light bg)
    text_primary="#1A1A1A",
    text_secondary="#404040",
    text_tertiary="#666666",
    text_quaternary="#999999",
    text_link="#0066CC",
    # Chart colors (same as dark for consistency)
    chart_blue="#0066CC",
    chart_green="#28A745",
    chart_orange="#E67E00",
    chart_red="#DC3545",
    chart_purple="#9B59B6",
    chart_cyan="#17A2B8",
    chart_yellow="#F0AD4E",
    chart_brown="#8B6914",
    # Grid and axis
    grid_line="#E0E0E0",
    axis_line="#C0C0C0",
    tick_color="#666666",
    # Shadows
    shadow_light="rgba(0, 0, 0, 0.08)",
    shadow_medium="rgba(0, 0, 0, 0.12)",
    shadow_heavy="rgba(0, 0, 0, 0.20)",
)


# High Contrast theme (accessibility)
HIGH_CONTRAST_THEME = ThemeColors(
    name="High Contrast",
    is_dark=True,
    # Primary accent - bright cyan for visibility
    primary="#00FFFF",
    primary_hover="#66FFFF",
    primary_pressed="#00CCCC",
    primary_muted="#00FFFF40",
    # Semantic colors (maximum contrast)
    success="#00FF00",
    success_hover="#66FF66",
    success_muted="#00FF0040",
    warning="#FFFF00",
    warning_hover="#FFFF66",
    warning_muted="#FFFF0040",
    error="#FF0000",
    error_hover="#FF6666",
    error_muted="#FF000040",
    # Background hierarchy (pure black base)
    bg_deep="#000000",
    bg_base="#000000",
    bg_surface="#1A1A1A",
    bg_elevated="#333333",
    bg_highlight="#4D4D4D",
    # Border colors (high contrast)
    border_subtle="#404040",
    border_default="#808080",
    border_strong="#FFFFFF",
    border_focus="#00FFFF",
    # Text hierarchy (white on black)
    text_primary="#FFFFFF",
    text_secondary="#FFFFFF",
    text_tertiary="#CCCCCC",
    text_quaternary="#999999",
    text_link="#00FFFF",
    # Chart colors (high contrast palette)
    chart_blue="#00FFFF",
    chart_green="#00FF00",
    chart_orange="#FFAA00",
    chart_red="#FF0000",
    chart_purple="#FF00FF",
    chart_cyan="#00FFFF",
    chart_yellow="#FFFF00",
    chart_brown="#FF8800",
    # Grid and axis
    grid_line="#404040",
    axis_line="#808080",
    tick_color="#FFFFFF",
    # Shadows (minimal for clarity)
    shadow_light="rgba(255, 255, 255, 0.1)",
    shadow_medium="rgba(255, 255, 255, 0.15)",
    shadow_heavy="rgba(255, 255, 255, 0.25)",
)


# Theme preset mapping
THEME_PRESETS: dict[ThemePreset, ThemeColors] = {
    ThemePreset.LIGHT: LIGHT_THEME,
    ThemePreset.DARK: DARK_THEME,
    ThemePreset.HIGH_CONTRAST: HIGH_CONTRAST_THEME,
}


class ThemeManager:
    """Singleton theme manager for application-wide theme control.

    Provides centralized theme management with support for:
    - Switching between Light, Dark, and High Contrast themes
    - Callback-based notifications for theme changes
    - Persistence of theme preference (via settings)

    Usage:
        manager = ThemeManager.instance()
        manager.set_theme(ThemePreset.DARK)
        colors = manager.colors
    """

    _instance: ClassVar["ThemeManager | None"] = None
    _theme_changed_callbacks: list[Callable[[ThemeColors], None]]

    _CUSTOM_THEME_KEY = "custom_themes"
    _SETTINGS_GROUP = "UpstreamDrift/Theme"

    def __init__(self) -> None:
        """Initialize the theme manager."""
        self._current_preset: ThemePreset | None = ThemePreset.DARK
        self._current_theme = DARK_THEME
        self._theme_changed_callbacks: list[Callable[[ThemeColors], None]] = []
        self._qt_signal: pyqtSignal | None = None
        self._fleet_theme_name: str | None = None
        self._custom_themes: dict[str, dict[str, str]] = {}
        self._load_custom_themes()

    @classmethod
    def instance(cls) -> "ThemeManager":
        """Get the singleton instance of ThemeManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def colors(self) -> ThemeColors:
        """Get current theme colors."""
        return self._current_theme

    @property
    def preset(self) -> ThemePreset | None:
        """Get current theme preset, or None if using a fleet theme."""
        return self._current_preset

    @property
    def is_dark(self) -> bool:
        """Check if current theme is dark."""
        return self._current_theme.is_dark

    def set_theme(self, preset: ThemePreset) -> None:
        """Set the current theme.

        Args:
            preset: Theme preset to apply

        Emits:
            theme_changed signal with new ThemeColors
        """
        if preset == self._current_preset and self._fleet_theme_name is None:
            return

        self._current_preset = preset
        self._current_theme = THEME_PRESETS[preset]
        self._fleet_theme_name = None  # Clear fleet theme when using preset

        self.save_theme_preference()

        # Notify all callbacks
        for callback in self._theme_changed_callbacks:
            callback(self._current_theme)

        # Emit Qt signal if available
        if self._qt_signal is not None:
            self._qt_signal.emit(self._current_theme)  # type: ignore[attr-defined]

    def on_theme_changed(self, callback: Callable[[ThemeColors], None]) -> None:
        """Register a callback for theme changes.

        Args:
            callback: Function to call when theme changes
        """
        if callback not in self._theme_changed_callbacks:
            self._theme_changed_callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ThemeColors], None]) -> None:
        """Remove a theme change callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._theme_changed_callbacks:
            self._theme_changed_callbacks.remove(callback)

    def set_fleet_theme(self, theme_name: str) -> None:
        """Set theme from the fleet-wide theme system.

        This allows using any of the 13 fleet-wide themes (Light, Dark,
        Slate Gray, Ocean Blue, Forest Green, Monokai, Dracula, One Dark,
        Gitpod Dark, MS Word, MS Excel, Legal Pad, High Contrast).

        Args:
            theme_name: Name of the fleet theme to apply

        Raises:
            KeyError: If theme_name is not found in fleet themes
            ImportError: If fleet theme system is not available
        """
        from .fleet_adapter import fleet_to_theme_colors, is_fleet_available

        if not is_fleet_available():
            raise ImportError(
                "Fleet theme system not available. "
                "Run 'git submodule update --init --recursive' to enable."
            )

        # Convert fleet theme to ThemeColors and apply
        self._current_theme = fleet_to_theme_colors(theme_name)
        self._current_preset = None  # Clear preset when using fleet theme
        self._fleet_theme_name = theme_name

        self.save_theme_preference()

        # Notify all callbacks
        for callback in self._theme_changed_callbacks:
            callback(self._current_theme)

        # Emit Qt signal if available
        if self._qt_signal is not None:
            self._qt_signal.emit(self._current_theme)  # type: ignore[attr-defined]

    def get_available_fleet_themes(self) -> list[str]:
        """Get list of available fleet-wide theme names.

        Returns:
            List of theme names, or empty list if fleet system unavailable
        """
        from .fleet_adapter import get_fleet_theme_names, is_fleet_available

        if not is_fleet_available():
            return []
        return get_fleet_theme_names()

    @property
    def theme_name(self) -> str:
        """Get the current theme name (works for both preset and fleet themes)."""
        if hasattr(self, "_fleet_theme_name") and self._fleet_theme_name:
            return self._fleet_theme_name
        return self._current_theme.name

    def get_stylesheet(self) -> str:
        """Generate a Qt stylesheet for the current theme.

        Returns:
            CSS-like stylesheet string for Qt widgets
        """
        c = self._current_theme
        return f"""
            QWidget {{
                background-color: {c.bg_base};
                color: {c.text_primary};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}

            QMainWindow {{
                background-color: {c.bg_deep};
            }}

            QDialog {{
                background-color: {c.bg_surface};
            }}

            QPushButton {{
                background-color: {c.primary};
                color: {c.text_primary};
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}

            QPushButton:hover {{
                background-color: {c.primary_hover};
            }}

            QPushButton:pressed {{
                background-color: {c.primary_pressed};
            }}

            QPushButton:disabled {{
                background-color: {c.bg_elevated};
                color: {c.text_quaternary};
            }}

            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {c.bg_surface};
                color: {c.text_primary};
                border: 1px solid {c.border_default};
                border-radius: 4px;
                padding: 4px 8px;
            }}

            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border-color: {c.border_focus};
            }}

            QComboBox {{
                background-color: {c.bg_surface};
                color: {c.text_primary};
                border: 1px solid {c.border_default};
                border-radius: 4px;
                padding: 4px 8px;
            }}

            QComboBox:hover {{
                border-color: {c.border_strong};
            }}

            QComboBox::drop-down {{
                border: none;
            }}

            QListWidget, QTreeWidget, QTableWidget {{
                background-color: {c.bg_surface};
                color: {c.text_primary};
                border: 1px solid {c.border_subtle};
            }}

            QListWidget::item:selected, QTreeWidget::item:selected {{
                background-color: {c.primary_muted};
            }}

            QScrollBar:vertical {{
                background-color: {c.bg_base};
                width: 12px;
            }}

            QScrollBar::handle:vertical {{
                background-color: {c.border_default};
                border-radius: 6px;
                min-height: 20px;
            }}

            QScrollBar::handle:vertical:hover {{
                background-color: {c.border_strong};
            }}

            QMenuBar {{
                background-color: {c.bg_surface};
                color: {c.text_primary};
            }}

            QMenuBar::item:selected {{
                background-color: {c.bg_elevated};
            }}

            QMenu {{
                background-color: {c.bg_surface};
                color: {c.text_primary};
                border: 1px solid {c.border_default};
            }}

            QMenu::item:selected {{
                background-color: {c.primary_muted};
            }}

            QTabWidget::pane {{
                border: 1px solid {c.border_default};
                background-color: {c.bg_surface};
            }}

            QTabBar::tab {{
                background-color: {c.bg_elevated};
                color: {c.text_secondary};
                padding: 8px 16px;
                border: none;
            }}

            QTabBar::tab:selected {{
                background-color: {c.bg_surface};
                color: {c.text_primary};
            }}

            QStatusBar {{
                background-color: {c.bg_surface};
                color: {c.text_tertiary};
            }}

            QLabel {{
                color: {c.text_primary};
            }}

            QGroupBox {{
                border: 1px solid {c.border_subtle};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}

            QGroupBox::title {{
                color: {c.text_secondary};
                subcontrol-origin: margin;
                left: 10px;
            }}

            QCheckBox::indicator, QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}

            QProgressBar {{
                background-color: {c.bg_elevated};
                border: none;
                border-radius: 4px;
            }}

            QProgressBar::chunk {{
                background-color: {c.primary};
                border-radius: 4px;
            }}

            QSlider::groove:horizontal {{
                background-color: {c.bg_elevated};
                height: 4px;
                border-radius: 2px;
            }}

            QSlider::handle:horizontal {{
                background-color: {c.primary};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}

            QToolTip {{
                background-color: {c.bg_surface};
                color: {c.text_primary};
                border: 1px solid {c.border_default};
                padding: 4px;
            }}
        """

    # ── Custom theme support ─────────────────────────────────────

    def _load_custom_themes(self) -> None:
        """Load custom themes from QSettings."""
        try:
            from PyQt6.QtCore import QSettings

            settings = QSettings("UpstreamDrift", "GolfModelingSuite")
            settings.beginGroup(self._SETTINGS_GROUP)
            raw = settings.value(self._CUSTOM_THEME_KEY, "")
            settings.endGroup()
            if raw:
                self._custom_themes = json.loads(raw)
        except Exception:
            self._custom_themes = {}

    def _save_custom_themes(self) -> None:
        """Persist custom themes to QSettings."""
        try:
            from PyQt6.QtCore import QSettings

            settings = QSettings("UpstreamDrift", "GolfModelingSuite")
            settings.beginGroup(self._SETTINGS_GROUP)
            settings.setValue(self._CUSTOM_THEME_KEY, json.dumps(self._custom_themes))
            settings.endGroup()
        except Exception as e:
            logger.warning("Failed to save custom themes: %s", e)

    def get_builtin_themes(self) -> list[str]:
        """Get list of built-in theme names (presets + fleet themes, deduplicated)."""
        seen: set[str] = set()
        names: list[str] = []
        for p in ThemePreset:
            n = p.name.replace("_", " ").title()
            if n not in seen:
                seen.add(n)
                names.append(n)
        for n in self.get_available_fleet_themes():
            if n not in seen:
                seen.add(n)
                names.append(n)
        return names

    def get_custom_theme_names(self) -> list[str]:
        """Get list of user-created custom theme names."""
        return list(self._custom_themes.keys())

    def get_available_themes(self) -> list[str]:
        """Get all theme names (built-in + custom)."""
        return self.get_builtin_themes() + self.get_custom_theme_names()

    def get_current_theme_name(self) -> str:
        """Get the name of the currently active theme."""
        return self.theme_name

    def get_theme_definition(self, name: str) -> dict[str, str] | None:
        """Get the color dictionary for a theme by name.

        Returns a dict of color keys → hex values suitable for the
        CustomThemeEditor, or None if not found.
        """
        # Custom themes are stored as dicts already
        if name in self._custom_themes:
            return dict(self._custom_themes[name])

        # Built-in: convert ThemeColors dataclass to dict
        theme_colors = self._resolve_theme_by_name(name)
        if theme_colors is None:
            return None

        return {
            "bg": theme_colors.bg_base,
            "group_bg": theme_colors.bg_surface,
            "border": theme_colors.border_default,
            "text": theme_colors.text_primary,
            "text_secondary": theme_colors.text_secondary,
            "label": theme_colors.text_tertiary,
            "focus": theme_colors.border_focus,
            "input_bg": theme_colors.bg_surface,
            "accent": theme_colors.primary,
            "title_bg": theme_colors.bg_elevated,
            "title_border": theme_colors.border_strong,
            "table_header": theme_colors.bg_elevated,
            "table_alt": theme_colors.bg_surface,
            "button_hover": theme_colors.primary_hover,
        }

    def _resolve_theme_by_name(self, name: str) -> ThemeColors | None:
        """Resolve a theme name to ThemeColors."""
        # Check presets
        for preset in ThemePreset:
            preset_name = preset.name.replace("_", " ").title()
            if preset_name == name:
                return THEME_PRESETS[preset]

        # Check fleet themes
        try:
            from .fleet_adapter import fleet_to_theme_colors, is_fleet_available

            if is_fleet_available():
                fleet_names = self.get_available_fleet_themes()
                if name in fleet_names:
                    return fleet_to_theme_colors(name)
        except Exception:
            pass

        return None

    def save_custom_theme(
        self,
        name: str,
        colors: dict[str, str],
        apply_immediately: bool = False,
    ) -> str:
        """Save a custom theme.

        Args:
            name: Theme name
            colors: Dict of color keys → hex values
            apply_immediately: If True, apply the theme after saving

        Returns:
            The saved theme name
        """
        self._custom_themes[name] = dict(colors)
        self._save_custom_themes()
        logger.info("Saved custom theme: %s", name)

        if apply_immediately:
            self.change_theme(name)

        return name

    def delete_custom_theme(self, name: str) -> bool:
        """Delete a custom theme by name."""
        if name not in self._custom_themes:
            return False
        del self._custom_themes[name]
        self._save_custom_themes()
        logger.info("Deleted custom theme: %s", name)
        return True

    def change_theme(self, name: str) -> None:
        """Change theme by name (works for preset, fleet, and custom themes).

        This is the unified entry point for theme switching from dialogs.
        """
        # Check presets first
        for preset in ThemePreset:
            preset_name = preset.name.replace("_", " ").title()
            if preset_name == name:
                self.set_theme(preset)
                return

        # Check fleet themes
        try:
            fleet_names = self.get_available_fleet_themes()
            if name in fleet_names:
                self.set_fleet_theme(name)
                return
        except Exception:
            pass

        # Custom theme
        if name in self._custom_themes:
            self._apply_custom_theme(name)
            return

        raise KeyError(f"Unknown theme: {name}")

    def _apply_custom_theme(self, name: str) -> None:
        """Apply a custom theme from stored color definitions."""
        colors = self._custom_themes.get(name)
        if not colors:
            raise KeyError(f"Custom theme not found: {name}")

        # Build a ThemeColors from the custom color dict with dark theme fallbacks
        base = DARK_THEME
        self._current_theme = ThemeColors(
            name=name,
            is_dark=True,  # Custom themes default to dark mode
            primary=colors.get("accent", base.primary),
            primary_hover=colors.get("button_hover", base.primary_hover),
            primary_pressed=base.primary_pressed,
            primary_muted=base.primary_muted,
            success=base.success,
            success_hover=base.success_hover,
            success_muted=base.success_muted,
            warning=base.warning,
            warning_hover=base.warning_hover,
            warning_muted=base.warning_muted,
            error=base.error,
            error_hover=base.error_hover,
            error_muted=base.error_muted,
            bg_deep=colors.get("bg", base.bg_deep),
            bg_base=colors.get("bg", base.bg_base),
            bg_surface=colors.get("group_bg", base.bg_surface),
            bg_elevated=colors.get("title_bg", base.bg_elevated),
            bg_highlight=base.bg_highlight,
            border_subtle=base.border_subtle,
            border_default=colors.get("border", base.border_default),
            border_strong=colors.get("title_border", base.border_strong),
            border_focus=colors.get("focus", base.border_focus),
            text_primary=colors.get("text", base.text_primary),
            text_secondary=colors.get("text_secondary", base.text_secondary),
            text_tertiary=colors.get("label", base.text_tertiary),
            text_quaternary=base.text_quaternary,
            text_link=colors.get("accent", base.text_link),
            chart_blue=base.chart_blue,
            chart_green=base.chart_green,
            chart_orange=base.chart_orange,
            chart_red=base.chart_red,
            chart_purple=base.chart_purple,
            chart_cyan=base.chart_cyan,
            chart_yellow=base.chart_yellow,
            chart_brown=base.chart_brown,
            grid_line=base.grid_line,
            axis_line=base.axis_line,
            tick_color=base.tick_color,
            shadow_light=base.shadow_light,
            shadow_medium=base.shadow_medium,
            shadow_heavy=base.shadow_heavy,
        )
        self._current_preset = None
        self._fleet_theme_name = None

        # Notify callbacks
        for callback in self._theme_changed_callbacks:
            callback(self._current_theme)
        if self._qt_signal is not None:
            self._qt_signal.emit(self._current_theme)  # type: ignore[attr-defined]

    def save_theme_preference(self) -> None:
        """Save current theme name to QSettings for submodule inheritance."""
        try:
            from PyQt6.QtCore import QSettings

            settings = QSettings("UpstreamDrift", "GolfModelingSuite")
            settings.beginGroup(self._SETTINGS_GROUP)
            settings.setValue("current_theme", self.theme_name)
            settings.endGroup()
        except Exception as e:
            logger.debug("Could not save theme preference: %s", e)

    def load_saved_theme(self) -> None:
        """Load and apply the previously saved theme preference."""
        try:
            from PyQt6.QtCore import QSettings

            settings = QSettings("UpstreamDrift", "GolfModelingSuite")
            settings.beginGroup(self._SETTINGS_GROUP)
            saved = settings.value("current_theme", "")
            settings.endGroup()
            if saved:
                self.change_theme(saved)
        except Exception as e:
            logger.debug("Could not load saved theme: %s", e)


# Convenience functions
def get_theme_manager() -> ThemeManager:
    """Get the singleton ThemeManager instance."""
    return ThemeManager.instance()


def get_current_colors() -> ThemeColors:
    """Get colors for the current theme."""
    return ThemeManager.instance().colors


def set_theme(preset: ThemePreset) -> None:
    """Set the application theme.

    Args:
        preset: Theme preset to apply
    """
    ThemeManager.instance().set_theme(preset)


__all__ = [
    "DARK_THEME",
    "HIGH_CONTRAST_THEME",
    "LIGHT_THEME",
    "THEME_PRESETS",
    "ThemeColors",
    "ThemeManager",
    "ThemePreset",
    "get_current_colors",
    "get_theme_manager",
    "set_theme",
]
