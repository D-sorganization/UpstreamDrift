# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""LabelledSlider: a slider widget with a label and formatted value display."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from movement_optimizer.gui.wheel_blocker import suppress_wheel_events


class LabelledSlider(QWidget):
    """Slider with label and formatted value display."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        lo: float,
        hi: float,
        default: float,
        unit: str,
        decimals: int = 1,
        steps: int = 200,
        tooltip: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if lo >= hi:
            raise ValueError(f"lo ({lo}) must be < hi ({hi})")
        self.lo = lo
        self.hi = hi
        self.decimals = decimals
        self.unit = unit

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        row = QHBoxLayout()
        self.name_label = QLabel(label)
        self.val_label = QLabel(self._fmt(default))
        self.val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.name_label)
        row.addStretch()
        row.addWidget(self.val_label)
        layout.addLayout(row)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName(label)
        self.name_label.setBuddy(self.slider)
        self.slider.setRange(0, steps)
        self.slider.setValue(self._to_tick(default))
        self.slider.valueChanged.connect(self._on_change)
        suppress_wheel_events(self.slider)
        layout.addWidget(self.slider)

        _tip = tooltip if tooltip else f"{label} ({lo:.{decimals}f}-{hi:.{decimals}f} {unit})"
        self.slider.setToolTip(_tip)
        self.name_label.setToolTip(_tip)

    def _fmt(self, val: float) -> str:
        return f"{val:.{self.decimals}f} {self.unit}"

    def _to_tick(self, val: float) -> int:
        frac = (val - self.lo) / (self.hi - self.lo)
        return int(frac * self.slider.maximum())

    def _from_tick(self, tick: int) -> float:
        frac = tick / self.slider.maximum()
        return self.lo + frac * (self.hi - self.lo)

    def _on_change(self, tick: int) -> None:
        val = self._from_tick(tick)
        self.val_label.setText(self._fmt(val))
        self.value_changed.emit(val)

    def value(self) -> float:
        return self._from_tick(self.slider.value())

    def set_value(self, val: float) -> None:
        self.slider.setValue(self._to_tick(val))
