from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QLabel
from upstream_drift_tools.ui.managers.unit_preferences_manager import (
    UnitPreferencesManager,
)
from upstream_drift_tools.ui.widgets.unit_aware_input import (
    UnitAwareDisplay,
    UnitAwareInput,
)


@pytest.fixture
def mock_prefs(monkeypatch) -> Any:
    prefs = UnitPreferencesManager()
    # Let's register a dummy category if needed, or just use existing ones
    # Length is usually a category. Let's assume 'length' exists.
    # We will mock the methods to be predictable.
    monkeypatch.setattr(
        prefs, "get_preferred_unit", lambda cat: "m" if cat == "length" else "custom"
    )
    monkeypatch.setattr(
        prefs,
        "convert_to_si",
        lambda val, cat, unit: val * 1000 if unit == "km" else val,
    )
    monkeypatch.setattr(
        prefs,
        "convert_from_si",
        lambda val, cat, unit: val / 1000 if unit == "km" else val,
    )

    with patch(
        "upstream_drift_tools.ui.widgets.unit_aware_input.get_unit_preferences_manager",
        return_value=prefs,
    ):
        yield prefs


def test_unit_aware_input_init(qapp, mock_prefs) -> None:
    widget = UnitAwareInput(category="length", label="Length:", show_label=True)
    assert widget._category == "length"
    assert widget._current_unit == "m"
    assert widget._value_input.value() == 0.0

    # Check children
    labels = widget.findChildren(QLabel)
    assert len(labels) == 1
    assert labels[0].text() == "Length:"


def test_unit_aware_input_value_changed(qapp, mock_prefs) -> None:
    widget = UnitAwareInput(category="length")
    widget._unit_combo.setCurrentText("km")  # Let's assume combo is not restrictive
    widget._current_unit = "km"

    mock_val_emit = MagicMock()
    mock_inp_emit = MagicMock()
    widget.value_changed.connect(mock_val_emit)
    widget.input_changed.connect(mock_inp_emit)

    widget._value_input.setValue(2.0)
    # 2 km -> 2000 m (SI)
    mock_val_emit.assert_called_with(2000.0)
    mock_inp_emit.assert_called_with(2000.0, "km")


def test_unit_aware_input_unit_changed(qapp, mock_prefs) -> None:
    widget = UnitAwareInput(category="length")
    widget._si_value = 5000.0  # 5000 m
    widget._current_unit = "m"

    mock_unit_emit = MagicMock()
    widget.unit_changed.connect(mock_unit_emit)

    widget._unit_combo.setCurrentText("km")
    widget._on_unit_changed("km")

    # 5000 m -> 5 km
    assert widget._value_input.value() == 5.0
    assert widget._current_unit == "km"
    mock_unit_emit.assert_called_with("km")


def test_unit_aware_input_setters(qapp, mock_prefs) -> None:
    widget = UnitAwareInput(category="length")
    widget.set_range(10.0, 100.0)
    assert widget._value_input.minimum() == 10.0
    assert widget._value_input.maximum() == 100.0

    widget.set_decimals(3)
    assert widget._value_input.decimals() == 3

    widget.set_readonly(True)
    assert widget._value_input.isReadOnly() is True
    assert widget._unit_combo.isEnabled() is False

    widget.set_unit("km")
    assert widget._unit_combo.currentText() == "km"


def test_unit_aware_input_set_value(qapp, mock_prefs) -> None:
    widget = UnitAwareInput(category="length")

    # set SI value
    widget.set_value(1000.0, is_si=True)
    assert widget.value_si() == 1000.0
    assert widget.value() == 1000.0  # because current unit is 'm'

    # set specific unit
    widget.set_value(5.0, unit="km")
    assert widget._current_unit == "km"
    assert widget.value_si() == 5000.0
    assert widget.value() == 5.0


def test_unit_aware_display_init(qapp, mock_prefs) -> None:
    disp = UnitAwareDisplay(category="length", label="Display:", show_label=True)
    assert disp._category == "length"
    assert disp._current_unit == "m"
    assert disp._value_label.text() == "0.00"


def test_unit_aware_display_set_value_si(qapp, mock_prefs) -> None:
    disp = UnitAwareDisplay(category="length")
    disp._current_unit = "km"
    disp.set_value_si(3000.0)  # 3000 m -> 3 km
    assert disp._value_label.text() == "3.00"


def test_unit_aware_display_unit_changed(qapp, mock_prefs) -> None:
    disp = UnitAwareDisplay(category="length")
    disp.set_value_si(4000.0)

    disp._on_unit_changed("km")
    assert disp._value_label.text() == "4.00"


def test_preference_changed_input(qapp, mock_prefs) -> None:
    widget = UnitAwareInput(category="length")
    widget._on_preference_changed("length", "km")
    assert widget._unit_combo.currentText() == "km"

    # ignoring other categories
    widget._on_preference_changed("weight", "kg")
    assert widget._unit_combo.currentText() == "km"


def test_preference_changed_display(qapp, mock_prefs) -> None:
    disp = UnitAwareDisplay(category="length")
    disp._on_preference_changed("length", "km")
    assert disp._unit_combo.currentText() == "km"
