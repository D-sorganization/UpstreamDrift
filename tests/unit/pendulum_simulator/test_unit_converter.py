"""Tests for src.shared.python.pendulum_simulator.gui.unit_converter (Issues #1949, #1744)."""

from __future__ import annotations

import math

import pytest
from src.shared.python.pendulum_simulator.gui.unit_converter import (
    UnitCategory,
    UnitPreferences,
    from_si,
    get_available_units,
    get_preset_names,
    get_unit_label,
    to_si,
)


class TestUnitCategory:
    def test_length_value(self) -> None:
        assert UnitCategory.LENGTH.value == "length"

    def test_angle_value(self) -> None:
        assert UnitCategory.ANGLE.value == "angle"

    def test_has_expected_categories(self) -> None:
        cats = [c.value for c in UnitCategory]
        assert "length" in cats
        assert "mass" in cats
        assert "torque" in cats
        assert "angle" in cats
        assert "force" in cats


class TestUnitPreferences:
    def test_default_construction_uses_si(self) -> None:
        prefs = UnitPreferences()
        assert prefs.get_unit(UnitCategory.LENGTH) == "m"

    def test_apply_preset_imperial(self) -> None:
        prefs = UnitPreferences()
        prefs.apply_preset("Imperial")
        assert prefs.get_unit(UnitCategory.LENGTH) == "in"
        assert prefs.get_unit(UnitCategory.ANGLE) == "deg"

    def test_apply_preset_engineering(self) -> None:
        prefs = UnitPreferences()
        prefs.apply_preset("Engineering")
        assert prefs.get_unit(UnitCategory.LENGTH) == "cm"
        assert prefs.get_unit(UnitCategory.ANGLE) == "deg"

    def test_invalid_preset_raises(self) -> None:
        prefs = UnitPreferences()
        with pytest.raises(AssertionError):
            prefs.apply_preset("NonExistent")

    def test_set_unit_valid(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.LENGTH, "cm")
        assert prefs.get_unit(UnitCategory.LENGTH) == "cm"

    def test_set_unit_invalid_raises(self) -> None:
        prefs = UnitPreferences()
        with pytest.raises(AssertionError):
            prefs.set_unit(UnitCategory.LENGTH, "km")  # not in options

    def test_set_and_get_angle_deg(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.ANGLE, "deg")
        assert prefs.get_unit(UnitCategory.ANGLE) == "deg"


class TestToSi:
    def test_si_length_identity(self) -> None:
        prefs = UnitPreferences()  # SI defaults
        assert to_si(1.0, UnitCategory.LENGTH, prefs) == pytest.approx(1.0)

    def test_cm_to_m(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.LENGTH, "cm")
        assert to_si(100.0, UnitCategory.LENGTH, prefs) == pytest.approx(1.0)

    def test_inches_to_m(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.LENGTH, "in")
        assert to_si(39.3701, UnitCategory.LENGTH, prefs) == pytest.approx(
            1.0, rel=1e-4
        )

    def test_deg_to_rad(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.ANGLE, "deg")
        result = to_si(180.0, UnitCategory.ANGLE, prefs)
        assert result == pytest.approx(math.pi, rel=1e-6)

    def test_lb_to_kg(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.MASS, "lb")
        result = to_si(2.204623, UnitCategory.MASS, prefs)
        assert result == pytest.approx(1.0, rel=1e-5)


class TestFromSi:
    def test_si_length_identity(self) -> None:
        prefs = UnitPreferences()  # SI defaults
        assert from_si(1.0, UnitCategory.LENGTH, prefs) == pytest.approx(1.0)

    def test_m_to_cm(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.LENGTH, "cm")
        assert from_si(1.0, UnitCategory.LENGTH, prefs) == pytest.approx(100.0)

    def test_rad_to_deg(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.ANGLE, "deg")
        result = from_si(math.pi, UnitCategory.ANGLE, prefs)
        assert result == pytest.approx(180.0, rel=1e-6)

    def test_unit_converter_round_trip(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.LENGTH, "in")
        value = 5.3
        assert to_si(
            from_si(value, UnitCategory.LENGTH, prefs), UnitCategory.LENGTH, prefs
        ) == pytest.approx(value, rel=1e-10)


class TestGetAvailableUnits:
    def test_unit_converter_returns_list(self) -> None:
        units = get_available_units(UnitCategory.LENGTH)
        assert isinstance(units, list)

    def test_length_contains_m(self) -> None:
        units = get_available_units(UnitCategory.LENGTH)
        assert "m" in units

    def test_angle_contains_rad_and_deg(self) -> None:
        units = get_available_units(UnitCategory.ANGLE)
        assert "rad" in units
        assert "deg" in units


class TestGetPresetNames:
    def test_unit_converter_returns_list(self) -> None:
        names = get_preset_names()
        assert isinstance(names, list)

    def test_includes_si(self) -> None:
        assert "SI" in get_preset_names()

    def test_includes_imperial(self) -> None:
        assert "Imperial" in get_preset_names()


class TestGetUnitLabel:
    def test_si_length_is_m(self) -> None:
        prefs = UnitPreferences()
        label = get_unit_label(UnitCategory.LENGTH, prefs)
        assert label == "m"

    def test_after_setting_cm(self) -> None:
        prefs = UnitPreferences()
        prefs.set_unit(UnitCategory.LENGTH, "cm")
        assert get_unit_label(UnitCategory.LENGTH, prefs) == "cm"
