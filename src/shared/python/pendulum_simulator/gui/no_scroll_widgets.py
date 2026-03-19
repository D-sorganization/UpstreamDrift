"""
Mousewheel-immune input widgets.

QDoubleSpinBox and QSpinBox respond to the mousewheel by default, which
causes accidental value changes when the user scrolls the controls panel.
These subclasses disable that behaviour while keeping all other functionality.

The mousewheel should ONLY affect navigation scrollbars — never value inputs.

Design by Contract
------------------
- Pre:  Parent widget must be a QWidget or None.
- Post: wheelEvent is a no-op on these widgets (event is ignored and propagated).
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSlider, QSpinBox, QWidget


class NoScrollSpinBox(QSpinBox):
    """QSpinBox that ignores mousewheel events."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: object) -> None:  # noqa: N802
        event.ignore()  # type: ignore[attr-defined]


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores mousewheel events."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: object) -> None:  # noqa: N802
        event.ignore()  # type: ignore[attr-defined]


class NoScrollSlider(QSlider):
    """QSlider that ignores mousewheel events.

    Note: Navigation sliders (like the animation timeline) should use
    the standard QSlider — this class is only for value-input sliders.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: object) -> None:  # noqa: N802
        event.ignore()  # type: ignore[attr-defined]


class NoScrollComboBox(QComboBox):
    """QComboBox that ignores mousewheel events."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: object) -> None:  # noqa: N802
        event.ignore()  # type: ignore[attr-defined]
