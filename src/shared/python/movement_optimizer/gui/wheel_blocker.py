# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Wheel event blocker for numeric inputs, sliders, and combo boxes.

Prevents mouse wheel events from accidentally changing values in editable
controls when the user is trying to scroll the surrounding page.
"""

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QWidget


class WheelEventBlocker(QObject):
    """Event filter that blocks wheel events and lets them propagate to parents."""

    def eventFilter(self, obj: QObject | None, event: QEvent | None) -> bool:
        if event is not None and event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(obj, event)


_blocker: WheelEventBlocker | None = None


def suppress_wheel_events(widget: QWidget) -> None:
    """Install a shared event filter to suppress wheel events on a widget.

    Args:
        widget: The Qt widget (e.g., QSlider, QComboBox) to protect from wheel scrolling.
    """
    global _blocker
    if _blocker is None:
        _blocker = WheelEventBlocker()

    widget.installEventFilter(_blocker)

    # In PyQt, QComboBox also needs its view to be protected sometimes, but the
    # combo box itself is the primary target for focus wheel events.
