"""Theme manager for Golf Modeling Suite.

Manages application-wide theme state and notifies components of theme changes.
"""

from typing import Any

from src.shared.python.engine_availability import PYQT6_AVAILABLE
from .colors import (
    DARK_PALETTE,
    HIGH_CONTRAST_PALETTE,
    LIGHT_PALETTE,
    set_active_palette,
)

if PYQT6_AVAILABLE:
    from PyQt6.QtCore import QObject, pyqtSignal
else:
    # Fallback stubs for non-GUI environments
    class QObject:  # type: ignore
        """Stub for QObject."""
        pass

    class pyqtSignal:  # type: ignore
        """Stub for pyqtSignal."""
        def __init__(self, *args: Any) -> None: pass
        def emit(self, *args: Any) -> None: pass
        def connect(self, func: Any) -> None: pass


class ThemeManager(QObject):
    """Singleton manager for application themes."""

    # Signal emitted when theme changes. Passes the new theme name.
    theme_changed = pyqtSignal(str)

    _instance = None

    def __init__(self) -> None:
        """Initialize the theme manager.

        Note: Use get_instance() to access the singleton.
        """
        if PYQT6_AVAILABLE:
            super().__init__()

        self._current_theme = "Light"

        # Ensure palette matches default
        set_active_palette(LIGHT_PALETTE)

    def set_theme(self, theme_name: str) -> None:
        """Set the active theme by name.

        Args:
            theme_name: One of 'Light', 'Dark', 'High Contrast'
        """
        if theme_name == self._current_theme:
            return

        if theme_name == "Light":
            set_active_palette(LIGHT_PALETTE)
        elif theme_name == "High Contrast":
            set_active_palette(HIGH_CONTRAST_PALETTE)
        else:
            # Default to Dark
            set_active_palette(DARK_PALETTE)
            theme_name = "Dark"

        self._current_theme = theme_name
        self.theme_changed.emit(theme_name)

    def apply_theme_to_app(self, app: Any) -> None:
        """Apply the current theme to the QApplication instance.

        Args:
            app: QApplication instance
        """
        if not PYQT6_AVAILABLE:
            return

        from .colors import Colors

        # Set Global Stylesheet
        # We use f-string to inject current colors
        app.setStyleSheet(f"""
            QMainWindow {{
                background-color: {Colors.BG_BASE};
                color: {Colors.TEXT_PRIMARY};
            }}
            QWidget {{
                background-color: {Colors.BG_BASE};
                color: {Colors.TEXT_PRIMARY};
                font-family: 'Segoe UI', sans-serif;
            }}
            QDialog {{
                background-color: {Colors.BG_BASE};
            }}
            QLabel {{
                color: {Colors.TEXT_PRIMARY};
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {Colors.BG_HIGHLIGHT};
                border: 1px solid {Colors.PRIMARY};
            }}
            QLineEdit {{
                background-color: {Colors.BG_SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 6px;
            }}
            QTextEdit {{
                background-color: {Colors.BG_SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER_DEFAULT};
            }}
            QGroupBox {{
                border: 1px solid {Colors.BORDER_DEFAULT};
                margin-top: 1.5ex;
                padding-top: 10px;
                border-radius: 4px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                color: {Colors.TEXT_SECONDARY};
            }}
            QComboBox {{
                background-color: {Colors.BG_ELEVATED};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-radius: 4px;
                padding: 5px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QTabWidget::pane {{
                border: 1px solid {Colors.BORDER_DEFAULT};
                background-color: {Colors.BG_BASE};
            }}
            QTabBar::tab {{
                background-color: {Colors.BG_SURFACE};
                color: {Colors.TEXT_SECONDARY};
                padding: 8px 12px;
                border: 1px solid {Colors.BORDER_DEFAULT};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.BG_BASE};
                color: {Colors.PRIMARY};
                font-weight: bold;
                border-bottom: 2px solid {Colors.PRIMARY};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
        """)

    def get_current_theme(self) -> str:
        """Get the name of the current theme."""
        return self._current_theme

    @classmethod
    def get_instance(cls) -> "ThemeManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
