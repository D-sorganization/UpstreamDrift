"""GUI utilities for eliminating duplication in PyQt6 applications.

This module re-exports all public symbols from the canonical
``src.shared.python.ui.qt.utils`` module.  Import from either path;
both resolve to the same implementation.

Usage::

    from src.shared.python.gui_pkg.gui_utils import (
        get_qapp,
        BaseApplicationWindow,
        create_dialog,
        setup_window_geometry,
    )
"""

from __future__ import annotations

# Canonical source — all implementation lives in ui.qt.utils
from src.shared.python.ui.qt.utils import (  # noqa: F401
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
    "apply_stylesheet",
    "create_button",
    "create_dialog",
    "create_label",
    "get_default_icon",
    "get_qapp",
    "setup_window_geometry",
]
