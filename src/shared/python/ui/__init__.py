"""UI components for Golf Modeling Suite.

This package provides reusable UI components used across the application:

- Toast notifications: Non-blocking, auto-dismissing messages
- Shortcuts overlay: Modal keyboard shortcut reference
- Loading buttons: Buttons with loading state indicators
- Preferences dialog: User settings interface
- Recent models panel: Quick access to recently used models

Usage:
    from shared.python.ui import ToastManager, ShortcutsOverlay, LoadingButton

    # Toast notifications
    toast_manager = ToastManager(main_window)
    toast_manager.show_success("Done!")

    # Keyboard shortcuts
    overlay = ShortcutsOverlay(main_window)
    overlay.show()

    # Loading buttons
    btn = LoadingButton("Launch")
    btn.set_loading(True, "Launching...")

    # Recent models
    panel = RecentModelsPanel()
    panel.model_selected.connect(on_select)
"""

from .loading_button import IconLoadingButton, LoadingButton, LoadingSpinner
from .preferences_dialog import PreferencesDialog, UserPreferences
from .recent_models import RecentModelItem, RecentModelsPanel
from .shortcuts_overlay import (
    DEFAULT_SHORTCUTS,
    Shortcut,
    ShortcutBadge,
    ShortcutsOverlay,
)
from .toast import Toast, ToastManager, ToastType

__all__ = [
    # Toast
    "Toast",
    "ToastManager",
    "ToastType",
    # Shortcuts
    "DEFAULT_SHORTCUTS",
    "Shortcut",
    "ShortcutBadge",
    "ShortcutsOverlay",
    # Loading Button
    "IconLoadingButton",
    "LoadingButton",
    "LoadingSpinner",
    # Preferences
    "PreferencesDialog",
    "UserPreferences",
    # Recent Models
    "RecentModelItem",
    "RecentModelsPanel",
]
