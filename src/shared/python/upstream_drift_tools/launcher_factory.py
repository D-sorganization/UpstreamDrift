"""Shared launcher factory for PyQt6 applications.

Provides a unified, DRY launcher pattern to replace the 30+ duplicate
``launch_pyqt6.py`` files scattered across tool directories.

Usage::

    from upstream_drift_tools.launcher_factory import (
        create_launcher_config,
        launch_app,
    )

    config = create_launcher_config(
        app_module="my_tool.main",
        window_title="My Tool",
        min_width=1024,
        min_height=768,
    )
    exit_code = launch_app(config, window_factory=MyMainWindow)

Addresses #763 (Phase 2: DRY consolidation).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─── Exceptions ──────────────────────────────────────────────────


class LauncherError(Exception):
    """Raised when launcher configuration or execution fails."""


# ─── Configuration ───────────────────────────────────────────────


@dataclass(frozen=True)
class LauncherConfig:
    """Configuration for launching a PyQt6 application.

    Attributes:
        app_module: Dotted module path for the application.
        window_title: Title to display on the main window.
        min_width: Minimum window width in pixels.
        min_height: Minimum window height in pixels.
        icon_path: Optional path to the window icon.
        extra: Additional key-value pairs for tool-specific config.
    """

    app_module: str
    window_title: str
    min_width: int = 800
    min_height: int = 600
    icon_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def create_launcher_config(
    app_module: str,
    window_title: str,
    min_width: int = 800,
    min_height: int = 600,
    icon_path: str | None = None,
    **extra: Any,
) -> LauncherConfig:
    """Create a validated launcher configuration.

    Args:
        app_module: Dotted module path for the application.
        window_title: Title for the main window.
        min_width: Minimum window width in pixels.
        min_height: Minimum window height in pixels.
        icon_path: Optional path to window icon.
        **extra: Additional tool-specific configuration.

    Returns:
        Frozen LauncherConfig dataclass.
    """
    return LauncherConfig(
        app_module=app_module,
        window_title=window_title,
        min_width=min_width,
        min_height=min_height,
        icon_path=icon_path,
        extra=extra,
    )


# ─── Validation (DbC Preconditions) ─────────────────────────────


def validate_launcher_config(config: LauncherConfig) -> None:
    """Validate launcher configuration (Design by Contract preconditions).

    Preconditions:
        - app_module is a non-empty string
        - window_title is a non-empty string
        - min_width >= 0
        - min_height >= 0

    Raises:
        LauncherError: If any precondition is violated.
    """
    if not config.app_module or not config.app_module.strip():
        raise LauncherError(
            "app_module must be a non-empty string identifying the application module"
        )
    if not config.window_title or not config.window_title.strip():
        raise LauncherError(
            "window_title must be a non-empty string for the window title bar"
        )
    if config.min_width < 0:
        raise LauncherError(f"min_width must be >= 0, got {config.min_width}")
    if config.min_height < 0:
        raise LauncherError(f"min_height must be >= 0, got {config.min_height}")


# ─── Internal Helpers ────────────────────────────────────────────


def _import_pyqt6() -> tuple[Any, type]:
    """Import PyQt6 and return (QApplication_instance, QMainWindow_class).

    Raises:
        ImportError: If PyQt6 is not installed.
    """
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    return app, QMainWindow


# ─── Launcher ────────────────────────────────────────────────────


def launch_app(
    config: LauncherConfig,
    window_factory: Callable[[], Any],
) -> int:
    """Launch a PyQt6 application using the given configuration.

    This is the single unified entry point that replaces all duplicate
    ``launch_pyqt6.py`` scripts across the codebase.

    Args:
        config: Validated launcher configuration.
        window_factory: Callable that creates the main window instance.

    Returns:
        Application exit code (0 for success, 1 for failure).

    Preconditions:
        - config passes validate_launcher_config()
        - window_factory is callable

    Postconditions:
        - Returns an integer exit code
    """
    if not (config is not None):
        raise ValueError("config must be provided")
    if not (config is not None):
        raise ValueError("config must be provided")
    validate_launcher_config(config)

    try:
        app, _ = _import_pyqt6()
    except ImportError as exc:
        logger.error(
            "PyQt6 is not installed. Cannot launch '%s': %s",
            config.window_title,
            exc,
        )
        return 1

    logger.info("Launching application: %s", config.window_title)

    try:
        window = window_factory()
        window.setWindowTitle(config.window_title)
        window.setMinimumSize(config.min_width, config.min_height)

        if config.icon_path:
            try:
                from PyQt6.QtGui import QIcon

                window.setWindowIcon(QIcon(config.icon_path))
            except (ImportError, OSError, ValueError) as icon_err:
                logger.warning("Could not set window icon: %s", icon_err)

        window.show()
        return int(app.exec())

    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error(
            "Failed to launch '%s': %s",
            config.window_title,
            exc,
            exc_info=True,
        )
        return 1
