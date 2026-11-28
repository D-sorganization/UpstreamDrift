"""Theme-manager lifecycle helper for the Cross-Engine Dashboard window."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable


def release_theme_manager(window: Any) -> None:
    """Drop the singleton ThemeManager's reference to a destroyed dashboard.

    ``apply_theme_to_window`` makes the first window it sees the manager's
    ``main_window``; without this, reopening the dashboard after closing it
    raises ``RuntimeError: wrapped C/C++ object ... has been deleted``.

    Design by Contract
    ------------------
    Post: the manager never retains ``window`` as its ``main_window``.
    """
    try:
        from src.shared.python.theme.theme_manager import ThemeManager

        manager = ThemeManager.instance()
    except ImportError:
        return
    if manager.main_window is window:
        manager.main_window = None


def dashboard_destroyed_slot(window: Any) -> Callable[..., None]:
    """Bind a dashboard window to a Qt-safe destroyed-signal callback."""
    return partial(_release_dashboard_theme_manager, window)


def _release_dashboard_theme_manager(window: Any, *_args: Any) -> None:
    """Ignore Qt's signal payload while releasing the captured window."""
    release_theme_manager(window)
