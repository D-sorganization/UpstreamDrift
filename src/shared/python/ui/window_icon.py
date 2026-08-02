"""Cross-platform window-icon and taskbar-identity helpers.

This module centralises the *correct* way to make a PyQt6 application show
its own icon — in the title bar **and** the Windows taskbar.

Why this exists
---------------
Calling ``window.setWindowIcon(...)`` alone is not enough on Windows. The
taskbar groups windows by an *Application User Model ID* (AppUserModelID).
When a Python GUI does not declare one, Windows attributes the window to the
``python.exe`` host process and shows the generic Python icon in the taskbar,
even though the title-bar icon is correct. The fix is to call
``SetCurrentProcessExplicitAppUserModelID`` **before** the first window is
shown, and to set the icon on *both* the ``QApplication`` and the window.

History: the icon repeatedly "regressed" because earlier fixes only adjusted
the icon file path/order and asserted ``windowIcon() is not None`` — which is
true even when the taskbar icon is wrong. The regression test for this module
asserts the AppUserModelID is actually set, which is the part that was missing.

Usage::

    from shared.python.ui import apply_window_icon

    apply_window_icon(
        app=QApplication.instance(),
        window=main_window,
        icon_candidates=[ASSETS_DIR / "app.ico", ASSETS_DIR / "app.png"],
        app_id="D-sorganization.MyApp",
    )
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

__all__ = [
    "apply_window_icon",
    "resolve_icon_path",
    "set_app_user_model_id",
]

_logger = logging.getLogger(__name__)


def _is_windows() -> bool:
    return sys.platform == "win32"


def resolve_icon_path(
    candidates: Iterable[str | Path],
) -> Path | None:
    """Return the first candidate path that exists on disk, or ``None``.

    Args:
        candidates: Ordered icon paths. ``.ico`` should precede ``.png`` on
            Windows because ``.ico`` carries the multi-resolution frames the
            taskbar needs.

    Returns:
        The first existing path as a :class:`~pathlib.Path`, or ``None`` when
        none of the candidates exist.

    Raises:
        TypeError: If *candidates* is ``None``.
    """
    if candidates is None:
        raise TypeError("candidates must be an iterable of paths, not None")
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def set_app_user_model_id(app_id: str) -> bool:
    """Declare an explicit Windows AppUserModelID for this process.

    This must be called *before* the first top-level window is shown so the
    Windows taskbar groups the window under the application's own icon rather
    than the host ``python.exe`` icon. It is a no-op on non-Windows platforms.

    Args:
        app_id: A non-empty identifier, conventionally dotted and unique to
            the application (e.g. ``"D-sorganization.UpstreamDrift"``).

    Returns:
        ``True`` if the identity was set (Windows only), ``False`` on other
        platforms or if the platform call failed.

    Raises:
        TypeError: If *app_id* is not a string.
        ValueError: If *app_id* is empty or whitespace-only.
    """
    if not isinstance(app_id, str):
        raise TypeError(f"app_id must be a string, got {type(app_id).__name__}")
    if not app_id.strip():
        raise ValueError("app_id must be a non-empty string")

    if not _is_windows():
        _logger.debug("set_app_user_model_id is a no-op on %s", sys.platform)
        return False

    try:
        import ctypes

        # ``windll`` only exists on Windows; cast to Any so mypy does not flag
        # the attribute on other platforms (a per-platform ``type: ignore``
        # would itself be reported as unused when checked on Windows).
        shell32 = cast(Any, ctypes).windll.shell32
        shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except (OSError, AttributeError) as exc:  # pragma: no cover - platform call
        _logger.warning("Could not set AppUserModelID %r: %s", app_id, exc)
        return False

    _logger.debug("AppUserModelID set to %r", app_id)
    return True


def apply_window_icon(
    *,
    app: Any,
    window: Any,
    icon_candidates: Sequence[str | Path],
    app_id: str | None = None,
    icon_factory: Callable[[str], Any] | None = None,
) -> Path | None:
    """Apply an icon to both the application and window, fixing the taskbar.

    Performs the full, correct sequence:

    1. Declares the Windows AppUserModelID (when *app_id* is given) so the
       taskbar uses the app's own icon.
    2. Resolves the first existing icon file from *icon_candidates*.
    3. Sets that icon on **both** the ``QApplication`` and the window — the
       application-level icon is what the taskbar reads.

    Args:
        app: The ``QApplication`` instance (or ``None`` to skip the app-level
            icon). Must expose ``setWindowIcon`` when not ``None``.
        window: The main window. Must expose ``setWindowIcon``.
        icon_candidates: Ordered icon paths; the first existing one wins.
        app_id: Optional Windows AppUserModelID. When provided,
            :func:`set_app_user_model_id` is called first.
        icon_factory: Callable building an icon object from a path string.
            Defaults to ``PyQt6.QtGui.QIcon``. Injectable for testing.

    Returns:
        The icon :class:`~pathlib.Path` that was applied, or ``None`` if no
        candidate existed (in which case no icon is set).

    Raises:
        TypeError: If *window* is ``None`` or lacks ``setWindowIcon``.
    """
    if window is None:
        raise TypeError("window must be provided")
    if not hasattr(window, "setWindowIcon"):
        raise TypeError("window must expose a setWindowIcon method")

    if app_id is not None:
        set_app_user_model_id(app_id)

    icon_path = resolve_icon_path(icon_candidates)
    if icon_path is None:
        _logger.warning(
            "No window icon found among %d candidate(s); taskbar icon may "
            "fall back to the host process icon",
            len(icon_candidates),
        )
        return None

    if icon_factory is None:
        from PyQt6.QtGui import QIcon

        icon_factory = QIcon

    icon = icon_factory(str(icon_path))
    if app is not None and hasattr(app, "setWindowIcon"):
        app.setWindowIcon(icon)
    window.setWindowIcon(icon)
    _logger.info("Applied window icon: %s", icon_path.name)
    return icon_path
