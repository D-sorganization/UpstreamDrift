"""Shared GUI widgets for physics engine applications.

Provides reusable UI primitives used across Drake, MuJoCo, and Pinocchio
engine frontends, eliminating cross-engine duplication.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6 import QtWidgets

try:
    from PyQt6 import QtWidgets as _QtWidgets
except ImportError:
    _QtWidgets = None  # type: ignore[assignment, misc]


class LogPanel(_QtWidgets.QTextEdit if _QtWidgets else object):  # type: ignore[misc]
    """Log panel widget for displaying diagnostic messages.

    A read-only, dark-themed text edit styled with a monosphere font
    suitable for simulation log output.
    """

    def __init__(self) -> None:
        """Initialize the log panel."""
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet(
            "background:#111; color:#0F0; font-family:Consolas; font-size:12px;"
        )


class SignalBlocker:
    """Context manager to temporarily block Qt signals for a set of widgets.

    Prevents circular signal/slot feedback during programmatic UI updates.

    Usage::

        with SignalBlocker(slider, spinbox):
            slider.setValue(50)
            spinbox.setValue(0.5)
    """

    def __init__(self, *widgets: QtWidgets.QWidget) -> None:
        """Initialize with widgets to block."""
        self.widgets = widgets

    def __enter__(self) -> SignalBlocker:
        """Block signals for all widgets."""
        for w in self.widgets:
            w.blockSignals(True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Restore signals for all widgets."""
        for w in self.widgets:
            w.blockSignals(False)
