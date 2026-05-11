"""Fleet-wide shared font management system.

This module provides a unified font system for all PyQt6 GUI applications
across the D-sorganization repository fleet, designed to emulate the "Claude Desktop"
professional feel.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from PyQt6.QtCore import QObject, QSettings, pyqtSignal
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


# Standard professional fonts mapped to Claude Desktop style
BUILTIN_FONTS = [
    "Inter",
    "Roboto",
    "Segoe UI",
    "Outfit",
    "Helvetica Neue",
    "Arial",
    "Sans Serif",
]


class FontManager(QObject):
    """Manages application-wide font settings with persistence.

    Like ThemeManager, this operates as a singleton per application
    and persists preferences using QSettings.
    """

    fontChanged = pyqtSignal(str)
    _instance: ClassVar[FontManager | None] = None

    def __init__(
        self,
        app_context: str | None = None,
        settings_org: str | None = None,
        settings_app: str | None = None,
    ) -> None:
        """Initialize the font manager.

        Args:
            app_context: Optional context name for scoped settings.
            settings_org: Organization name for QSettings.
            settings_app: Application name for QSettings.
        """
        super().__init__()
        self.app_context = app_context or "Global"

        # Configure settings scope
        if settings_org and settings_app:
            self.settings = QSettings(settings_org, settings_app)
        else:
            self.settings = QSettings("D-sorganization", "FleetShared")

        self.settings.beginGroup(f"Font_{self.app_context}")

        self.current_font = self._load_preference()
        logger.info(
            f"FontManager initialized: font={self.current_font}, context={self.app_context}"
        )

    def _load_preference(self) -> str:
        """Load the saved font preference or return the default."""
        saved_font = str(self.settings.value("font_family", "Inter", type=str))
        if saved_font not in BUILTIN_FONTS and saved_font != "System Default":
            # Just in case the font isn't available, but we allow system fonts too.
            pass
        return saved_font

    def get_available_fonts(self) -> list[str]:
        """Return a list of available built-in professional fonts."""
        # Check which of the built-in fonts are actually available on the system
        db = QFontDatabase()
        system_fonts = db.families()

        available = [f for f in BUILTIN_FONTS if f in system_fonts]
        # Always add a system default option
        if "System Default" not in available:
            available.append("System Default")

        # If no professional fonts are found, fallback
        if not available:
            available = ["System Default", "Arial"]

        return available

    def get_current_font(self) -> str:
        """Return the currently selected font family name."""
        return self.current_font

    def change_font(self, font_family: str) -> None:
        """Change the global font and emit a signal.

        Args:
            font_family: Name of the font family to apply.
        """
        if font_family == self.current_font:
            return

        self.current_font = font_family
        self.settings.setValue("font_family", font_family)

        self.apply_font()
        self.fontChanged.emit(font_family)
        logger.info(f"Font changed to: {font_family}")

    def apply_font(self, app: QApplication | None = None) -> None:
        """Apply the current font to the application.

        Args:
            app: Optional QApplication instance. If None, uses QApplication.instance().
        """
        target_app = app or QApplication.instance()
        if target_app is None:
            logger.warning("No QApplication instance available to apply font.")
            return

        if self.current_font == "System Default":
            # Revert to Qt's default system font
            target_app.setFont(QFont())
            return

        font = QFont(self.current_font)
        # Professional UI standard sizing
        font.setPointSize(10)
        target_app.setFont(font)


def get_font_manager(
    app_context: str | None = None,
    settings_org: str | None = None,
    settings_app: str | None = None,
) -> FontManager:
    """Get or create the singleton FontManager instance.

    Args:
        app_context: Optional context name for scoped settings.
        settings_org: Organization name for QSettings.
        settings_app: Application name for QSettings.

    Returns:
        The singleton FontManager instance.
    """
    if FontManager._instance is None:
        FontManager._instance = FontManager(
            app_context=app_context,
            settings_org=settings_org,
            settings_app=settings_app,
        )
    return FontManager._instance
