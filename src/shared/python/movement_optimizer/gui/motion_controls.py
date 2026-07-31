# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Reusable PyQt controls for movement-optimizer motion tabs."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QWidget,
)


class NumericControl(QWidget):
    """Slider plus typed value field without spin-box arrows."""

    valueChanged = pyqtSignal(float)  # noqa: N815 - Qt signal naming convention.

    def __init__(
        self,
        lower: float,
        upper: float,
        value: float,
        *,
        integer: bool = False,
        decimals: int = 3,
        steps: int = 1000,
    ) -> None:
        super().__init__()
        if upper <= lower:
            raise ValueError("upper must be greater than lower")
        self._lower = lower
        self._upper = upper
        self._integer = integer
        self._decimals = 0 if integer else decimals
        self._steps = max(1, int(upper - lower) if integer else steps)
        self._value = self._coerce(value)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.setMinimumHeight(32)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setTracking(False)
        self.slider.setMinimumHeight(28)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.edit = QLineEdit()
        self.edit.setFixedWidth(88)
        self.edit.setMinimumHeight(28)
        self.edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.edit)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.edit.editingFinished.connect(self._on_text_changed)
        self._sync_widgets()

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        new_value = self._coerce(value)
        if new_value == self._value:
            self._sync_widgets()
            return
        self._value = new_value
        self._sync_widgets()
        self.valueChanged.emit(self._value)

    def _coerce(self, value: float) -> float:
        bounded = min(max(float(value), self._lower), self._upper)
        return float(round(bounded)) if self._integer else bounded

    def _value_to_slider(self, value: float) -> int:
        ratio = (value - self._lower) / (self._upper - self._lower)
        return round(ratio * self._steps)

    def _slider_to_value(self, position: int) -> float:
        ratio = position / self._steps
        return self._coerce(self._lower + ratio * (self._upper - self._lower))

    def _sync_widgets(self) -> None:
        slider_value = self._value_to_slider(self._value)
        if self.slider.value() != slider_value:
            self.slider.blockSignals(True)
            self.slider.setValue(slider_value)
            self.slider.blockSignals(False)
        text = (
            f"{int(self._value)}"
            if self._integer
            else f"{self._value:.{self._decimals}f}"
        )
        if self.edit.text() != text:
            self.edit.setText(text)

    def _on_slider_changed(self, position: int) -> None:
        self._value = self._slider_to_value(position)
        self._sync_widgets()
        self.valueChanged.emit(self._value)

    def _on_text_changed(self) -> None:
        try:
            parsed = float(self.edit.text())
        except ValueError:
            self._sync_widgets()
            return
        self.set_value(parsed)


def scrollable_control_panel(panel: QWidget) -> QScrollArea:
    scroll_area = QScrollArea()
    scroll_area.setWidget(panel)
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll_area.setMinimumWidth(340)
    scroll_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    return scroll_area
