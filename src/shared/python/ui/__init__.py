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

from typing import Any

from .window_icon import (
    apply_window_icon,
    resolve_icon_path,
    set_app_user_model_id,
)

# PyQt6-dependent imports - only available when PyQt6 is installed.
# Guarded so the package imports cleanly in headless environments without
# PyQt6 (mirrors theme/__init__), exposing the Qt widgets only when present.
_AutoCompleteLineEdit: Any = None
_HoverCopyTextBrowser: Any = None
_LoadingButton: Any = None
_ShortcutsOverlay: Any = None
_ToastManager: Any = None
try:
    from .auto_complete import AutoCompleteLineEdit as _ImportedAutoCompleteLineEdit
    from .hover_copy_browser import (
        HoverCopyTextBrowser as _ImportedHoverCopyTextBrowser,
    )
    from .loading_button import LoadingButton as _ImportedLoadingButton
    from .shortcuts_overlay import ShortcutsOverlay as _ImportedShortcutsOverlay
    from .toast import ToastManager as _ImportedToastManager

    _PYQT6_AVAILABLE = True
    _AutoCompleteLineEdit = _ImportedAutoCompleteLineEdit
    _HoverCopyTextBrowser = _ImportedHoverCopyTextBrowser
    _LoadingButton = _ImportedLoadingButton
    _ShortcutsOverlay = _ImportedShortcutsOverlay
    _ToastManager = _ImportedToastManager
except ImportError:
    _PYQT6_AVAILABLE = False

AutoCompleteLineEdit: Any = _AutoCompleteLineEdit
HoverCopyTextBrowser: Any = _HoverCopyTextBrowser
LoadingButton: Any = _LoadingButton
ShortcutsOverlay: Any = _ShortcutsOverlay
ToastManager: Any = _ToastManager

__all__ = [
    # Auto Complete
    "AutoCompleteLineEdit",
    # Hover Copy Text Browser
    "HoverCopyTextBrowser",
    # Toast notifications, shortcuts overlay, loading buttons (issue #8972)
    "LoadingButton",
    "ShortcutsOverlay",
    "ToastManager",
    # Window icon / taskbar identity
    "apply_window_icon",
    "resolve_icon_path",
    "set_app_user_model_id",
]
