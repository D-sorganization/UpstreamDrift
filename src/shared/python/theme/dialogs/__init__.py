"""Shared custom theme dialogs for PyQt6 applications.

Provides reusable dialogs for creating, editing, and managing custom themes.
These dialogs work with the shared ThemeManager to persist custom themes.

Usage:
    from shared.python.theme.dialogs import CustomThemeEditor, ThemeManagerDialog

    editor = CustomThemeEditor(theme_manager, parent)
    editor.exec()
"""

from .custom_theme_dialog import ColorFieldEditor, CustomThemeDialog
from .custom_theme_editor import (
    ColorPickerButton,
    CustomThemeEditor,
    ThemePreviewWidget,
)
from .theme_manager_dialog import ThemeListItem, ThemeManagerDialog

__all__ = [
    "ColorFieldEditor",
    "ColorPickerButton",
    "CustomThemeDialog",
    "CustomThemeEditor",
    "ThemeListItem",
    "ThemeManagerDialog",
    "ThemePreviewWidget",
]
