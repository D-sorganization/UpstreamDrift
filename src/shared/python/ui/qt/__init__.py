"""Shared Qt UI utilities package.

Provides reusable Qt widgets, application helpers, and a process-worker thread
used across all GUI-bearing engines and launchers.
"""

from .plotting import MplCanvas
from .process_worker import ProcessWorker
from .utils import (
    BaseApplicationWindow,
    LayoutBuilder,
    apply_stylesheet,
    create_button,
    create_dialog,
    create_label,
    get_default_icon,
    get_qapp,
    setup_window_geometry,
)

__all__ = [
    "BaseApplicationWindow",
    "LayoutBuilder",
    "MplCanvas",
    "ProcessWorker",
    "apply_stylesheet",
    "create_button",
    "create_dialog",
    "create_label",
    "get_default_icon",
    "get_qapp",
    "setup_window_geometry",
]
