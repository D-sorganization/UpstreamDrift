"""Open one tool widget as its own desktop window (``python -m src.tools.<tool>``).

Several tools ship a ``__main__`` whose only job is to start a
``QApplication``, show the tool's main widget and run the event loop, with a
plain install hint when the optional GUI extra is missing. This module holds
that sequence once so each tool's entry point is a single call (#14177).

PyQt6 and the tool's GUI module are imported inside the call, so importing
this module (or a tool's ``__main__``) never needs the GUI extra.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

_INSTALL_HINT = (
    "The desktop window requires the GUI extra. Install with:\n"
    "  pip install -e '.[gui-tools]'\n"
)


def _application() -> Any:
    """Return the running ``QApplication`` or start one for this process."""
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def _parse_widget_path(widget_path: str) -> tuple[str, str]:
    module_name, _, class_name = widget_path.partition(":")
    if not module_name or not class_name:
        raise ValueError(
            f"widget_path must look like 'module:ClassName', got {widget_path!r}"
        )
    return module_name, class_name


def run_standalone_widget(widget_path: str, size: tuple[int, int]) -> int:
    """Show ``widget_path`` in its own window and run the Qt event loop.

    Args:
        widget_path: ``"package.module:ClassName"`` of a widget class whose
            constructor takes no required arguments.
        size: Initial window ``(width, height)`` in pixels; both positive.

    Returns:
        The event loop's exit code, or ``1`` (with an install hint on
        stderr) when PyQt6 or the tool's GUI module cannot be imported.

    Raises:
        ValueError: If ``widget_path`` has no class name or ``size`` is not
            two positive integers.
    """
    module_name, class_name = _parse_widget_path(widget_path)
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive (width, height), got {size!r}")
    try:
        widget_class = getattr(importlib.import_module(module_name), class_name)
        app = _application()
    except (ImportError, OSError) as exc:
        sys.stderr.write(f"{_INSTALL_HINT}Import error: {exc}\n")
        return 1
    widget = widget_class()
    widget.resize(width, height)
    widget.show()
    return int(app.exec())


__all__ = ["run_standalone_widget"]
