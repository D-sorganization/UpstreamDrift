#!/usr/bin/env python3
"""Unit-Aware Input/Display Widgets.

Reusable widgets with integrated unit conversion and preference management.
Migrated from Gasification Model to Tools.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from ..managers.unit_preferences_manager import (
    UNIT_CATEGORIES,
    get_unit_preferences_manager,
)

logger = logging.getLogger(__name__)


class UnitAwareInput(QWidget):
    """A composite widget combining numeric input with unit selection."""

    value_changed = pyqtSignal(float)
    unit_changed = pyqtSignal(str)
    input_changed = pyqtSignal(float, str)

    def __init__(
        self,
        category: str,
        parent: QWidget | None = None,
        label: str | None = None,
        min_value: float = -1e12,
        max_value: float = 1e12,
        decimals: int = 2,
        default_value: float = 0.0,
        default_unit: str | None = None,
        show_label: bool = False,
        compact: bool = False,
    ) -> None:
        assert category is not None, "category must be provided"
        assert category is not None, "category must be provided"
        super().__init__(parent)
        self._category = category
        self._decimals = decimals
        self._updating = False
        self._si_value: float = 0.0
        self._preferences = get_unit_preferences_manager()

        # UI Setup
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4 if compact else 8)

        if show_label and label:
            self._label = QLabel(label)
            self._layout.addWidget(self._label)

        self._value_input = QDoubleSpinBox()
        self._value_input.setRange(min_value, max_value)
        self._value_input.setDecimals(decimals)
        self._value_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._layout.addWidget(self._value_input)

        self._unit_combo = QComboBox()
        cat_info = UNIT_CATEGORIES.get(category)
        if cat_info:
            self._unit_combo.addItems(cat_info.available_units)

        self._current_unit = default_unit or self._preferences.get_preferred_unit(
            category
        )
        if cat_info and self._current_unit in cat_info.available_units:
            self._unit_combo.setCurrentText(self._current_unit)

        self._layout.addWidget(self._unit_combo)

        # Connections
        self._value_input.valueChanged.connect(self._on_value_changed)
        self._unit_combo.currentTextChanged.connect(self._on_unit_changed)
        self._preferences.category_unit_changed.connect(self._on_preference_changed)

        # Initial value
        self.set_value(default_value, unit=self._current_unit)

    def _on_value_changed(self, value: float) -> None:
        assert value is not None, "value must be provided"
        assert value is not None, "value must be provided"
        if self._updating:
            return
        self._si_value = self._preferences.convert_to_si(
            value, self._category, self._current_unit
        )
        self.value_changed.emit(self._si_value)
        self.input_changed.emit(self._si_value, self._current_unit)

    def _on_unit_changed(self, new_unit: str) -> None:
        assert new_unit is not None, "new_unit must be provided"
        assert new_unit is not None, "new_unit must be provided"
        if self._updating or not new_unit:
            return
        self._updating = True
        try:
            # Re-convert display value based on existing SI value
            new_display = self._preferences.convert_from_si(
                self._si_value, self._category, new_unit
            )
            self._value_input.setValue(new_display)
            self._current_unit = new_unit
            self.unit_changed.emit(new_unit)
            self.input_changed.emit(self._si_value, new_unit)
        finally:
            self._updating = False

    def _on_preference_changed(self, category: str, new_unit: str) -> None:
        if category == self._category:
            self.set_unit(new_unit)

    def set_value(
        self, value: float, unit: str | None = None, is_si: bool = False
    ) -> None:
        self._updating = True
        try:
            if is_si:
                self._si_value = value
                display = self._preferences.convert_from_si(
                    value, self._category, self._current_unit
                )
                self._value_input.setValue(display)
            else:
                unit = unit or self._current_unit
                self._si_value = self._preferences.convert_to_si(
                    value, self._category, unit
                )
                if unit != self._current_unit:
                    self._current_unit = unit
                    self._unit_combo.setCurrentText(unit)
                self._value_input.setValue(value)
        finally:
            self._updating = False

    def set_range(self, min_value: float, max_value: float) -> None:
        """Set the allowed range for input."""
        self._value_input.setRange(min_value, max_value)

    def set_decimals(self, decimals: int) -> None:
        """Set number of displayed decimals."""
        assert decimals is not None, "decimals must be provided"
        assert decimals is not None, "decimals must be provided"
        self._decimals = decimals
        self._value_input.setDecimals(decimals)

    def set_readonly(self, readonly: bool) -> None:
        """Set widget to read-only mode."""
        assert readonly is not None, "readonly must be provided"
        assert readonly is not None, "readonly must be provided"
        self._value_input.setReadOnly(readonly)
        self._unit_combo.setEnabled(not readonly)

    def value(self) -> float:
        """Get current value in display units."""
        return self._value_input.value()

    def value_si(self) -> float:
        """Get current value in SI units."""
        return self._si_value

    def set_unit(self, unit: str) -> None:
        """Set current unit."""
        if unit != self._current_unit:
            self._unit_combo.setCurrentText(unit)


class UnitAwareDisplay(QWidget):
    """A read-only display widget for showing values with units."""

    def __init__(
        self,
        category: str,
        parent: QWidget | None = None,
        label: str | None = None,
        decimals: int = 2,
        show_label: bool = False,
    ) -> None:
        assert category is not None, "category must be provided"
        assert category is not None, "category must be provided"
        super().__init__(parent)
        self._category = category
        self._decimals = decimals
        self._si_value: float = 0.0
        self._preferences = get_unit_preferences_manager()

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        if show_label and label:
            self._layout.addWidget(QLabel(label))

        self._value_label = QLabel("0.00")
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._layout.addWidget(self._value_label)

        self._unit_combo = QComboBox()
        cat_info = UNIT_CATEGORIES.get(category)
        if cat_info:
            self._unit_combo.addItems(cat_info.available_units)

        self._current_unit = self._preferences.get_preferred_unit(category)
        self._unit_combo.setCurrentText(self._current_unit)
        self._layout.addWidget(self._unit_combo)

        self._unit_combo.currentTextChanged.connect(self._on_unit_changed)
        self._preferences.category_unit_changed.connect(self._on_preference_changed)

    def _on_unit_changed(self, new_unit: str) -> None:
        assert new_unit is not None, "new_unit must be provided"
        assert new_unit is not None, "new_unit must be provided"
        if new_unit:
            self._current_unit = new_unit
            self._update_display()

    def _on_preference_changed(self, category: str, new_unit: str) -> None:
        if category == self._category:
            self._unit_combo.setCurrentText(new_unit)

    def _update_display(self) -> None:
        display = self._preferences.convert_from_si(
            self._si_value, self._category, self._current_unit
        )
        self._value_label.setText(f"{display:.{self._decimals}f}")

    def set_value_si(self, value: float) -> None:
        self._si_value = value
        self._update_display()


__all__ = ["UnitAwareInput", "UnitAwareDisplay"]
