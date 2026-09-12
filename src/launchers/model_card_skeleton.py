"""Skeleton placeholder card widget for launcher loading state (issue #8906)."""

from __future__ import annotations

from PyQt6.QtCore import pyqtProperty  # type: ignore[attr-defined]
from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QSequentialAnimationGroup,
)
from PyQt6.QtGui import QHideEvent
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
)


class SkeletonCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SkeletonCard")
        self.setMinimumSize(180, 240)
        self.setStyleSheet("""
            #SkeletonCard {
                background-color: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
            }
        """)

        # Qt only allows one QGraphicsEffect per widget, and a real pulse
        # needs a QGraphicsOpacityEffect (setWindowOpacity() is a no-op on
        # a non top-level widget -- issue #8906), so the opacity effect
        # replaces the drop shadow as the card's active effect.
        self.effect = QGraphicsOpacityEffect(self)
        self.effect.setOpacity(0.35)
        self.setGraphicsEffect(self.effect)

        self._pulse_opacity: float = 0.35

        pulse_up = QPropertyAnimation(self, b"pulseOpacity")
        pulse_up.setDuration(1000)
        pulse_up.setStartValue(0.35)
        pulse_up.setEndValue(0.85)
        pulse_up.setEasingCurve(QEasingCurve.Type.InOutSine)

        pulse_down = QPropertyAnimation(self, b"pulseOpacity")
        pulse_down.setDuration(1000)
        pulse_down.setStartValue(0.85)
        pulse_down.setEndValue(0.35)
        pulse_down.setEasingCurve(QEasingCurve.Type.InOutSine)

        self._anim = QSequentialAnimationGroup(self)
        self._anim.addAnimation(pulse_up)
        self._anim.addAnimation(pulse_down)
        self._anim.setLoopCount(-1)
        self._anim.start()

    @pyqtProperty(float)
    def pulseOpacity(self) -> float:
        return self._pulse_opacity

    @pulseOpacity.setter  # type: ignore[no-redef]
    def pulseOpacity(self, value: float) -> None:
        self._pulse_opacity = value
        self.effect.setOpacity(value)

    def hideEvent(self, event: QHideEvent | None) -> None:
        """Stop the pulse animation once the skeleton is hidden.

        Without this, `_anim`'s loop-forever animation keeps running after
        the card is torn down (e.g. by `_rebuild_grid`'s grid-teardown
        loop), leaking a timer for the lifetime of the process (#8906).
        """
        self._anim.stop()
        super().hideEvent(event)
