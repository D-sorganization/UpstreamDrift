# ruff: noqa: E501
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QSettings
from sidekick.ui.managers.unit_preferences_manager import (
    UnitPreferencesManager,
    _PreferencesHolder,
    get_unit_preferences_manager,
)


@pytest.fixture(autouse=True)
def clean_preferences() -> Any:
    # Clear the singleton before and after each test
    _PreferencesHolder.instance = None
    settings = QSettings("UpstreamDriftTools", "UnitPreferences")
    settings.clear()
    yield
    _PreferencesHolder.instance = None
    settings.clear()


def test_manager_singleton() -> Any:
    manager1 = get_unit_preferences_manager()
    manager2 = get_unit_preferences_manager()
    assert manager1 is manager2


def test_load_preferences_default(qapp) -> Any:
    manager = UnitPreferencesManager()
    assert manager._preset_name == "Default"
    assert manager.get_preferred_unit("temperature") == "°C"


def test_load_preferences_custom(qapp) -> Any:
    settings = QSettings("UpstreamDriftTools", "UnitPreferences")
    custom_prefs = {"temperature": "K"}
    settings.setValue("unit_preferences", json.dumps(custom_prefs))
    settings.setValue("preset_name", "Custom")

    manager = UnitPreferencesManager()
    assert manager._preset_name == "Custom"
    assert manager.get_preferred_unit("temperature") == "K"
    assert manager.get_preferred_unit("pressure") == "atm"  # fell back to default


def test_load_preferences_bad_json(qapp) -> Any:
    settings = QSettings("UpstreamDriftTools", "UnitPreferences")
    settings.setValue("unit_preferences", "{bad json}")
    manager = UnitPreferencesManager()
    assert manager.get_preferred_unit("temperature") == "°C"  # default recovery


def test_set_preferred_unit(qapp) -> Any:
    manager = UnitPreferencesManager()
    settings = QSettings("UpstreamDriftTools", "UnitPreferences")

    def on_changed(cat, unit) -> Any:
        assert cat == "temperature"
        assert unit == "K"

    manager.category_unit_changed.connect(on_changed)

    manager.set_preferred_unit("temperature", "K")
    assert manager.get_preferred_unit("temperature") == "K"

    # Verify it was saved
    saved_str = settings.value("unit_preferences", "{}")
    saved = json.loads(saved_str if isinstance(saved_str, str) else "{}")
    assert saved.get("temperature") == "K"


def test_set_preferred_unit_invalid(qapp) -> Any:
    manager = UnitPreferencesManager()
    manager.set_preferred_unit("invalid_category", "X")
    manager.set_preferred_unit("temperature", "invalid_unit")
    assert manager.get_preferred_unit("temperature") == "°C"


def test_get_si_unit() -> Any:
    manager = UnitPreferencesManager()
    assert manager.get_si_unit("temperature") == "K"
    assert manager.get_si_unit("invalid_category") == ""


@patch("sidekick.ui.managers.unit_preferences_manager.UnitPreferencesManager.converter")
def test_convert_to_si(mock_converter, qapp) -> Any:
    manager = UnitPreferencesManager()
    # Mock the return of converter.convert().value
    mock_convert_result = MagicMock()
    mock_convert_result.value = 273.15
    mock_converter.convert.return_value = mock_convert_result

    # Base case
    res = manager.convert_to_si(0.0, "temperature", "°C")
    assert res == 273.15
    mock_converter.convert.assert_called_once_with(0.0, "°C", "K")

    # When from_unit == si_unit
    assert manager.convert_to_si(100.0, "temperature", "K") == 100.0


@patch("sidekick.ui.managers.unit_preferences_manager.UnitPreferencesManager.converter")
def test_convert_from_si(mock_converter, qapp) -> Any:
    manager = UnitPreferencesManager()
    mock_convert_result = MagicMock()
    mock_convert_result.value = 0.0
    mock_converter.convert.return_value = mock_convert_result

    res = manager.convert_from_si(273.15, "temperature", "°C")
    assert res == 0.0
    mock_converter.convert.assert_called_once_with(273.15, "K", "°C")

    # When to_unit == si_unit
    assert manager.convert_from_si(100.0, "temperature", "K") == 100.0


@patch("sidekick.ui.managers.unit_preferences_manager.UnitPreferencesManager.converter")
def test_convert_error_handling(mock_converter, qapp) -> Any:
    manager = UnitPreferencesManager()
    mock_converter.convert.side_effect = ValueError("Bad unit")

    # Should fall back to returning original value
    assert manager.convert_to_si(123.4, "temperature", "°C") == 123.4
    assert manager.convert_from_si(567.8, "temperature", "°C") == 567.8


def test_lazy_converter_property(qapp) -> Any:
    # Tests the lazy property initialization
    manager = UnitPreferencesManager()
    with patch(
        "sidekick.calculators.conversion.service.get_service",
        return_value="MOCKED_SERVICE",
    ):
        assert manager.converter == "MOCKED_SERVICE"
        # Accessing it again should not re-call get_service
        assert manager.converter == "MOCKED_SERVICE"
