#!/usr/bin/env python3
"""Unit Preferences Manager - Global unit preferences management for the application.

Shared component for managing user unit preferences across the fleet.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QSettings, pyqtSignal

if TYPE_CHECKING:
    from ...calculators.conversion.service import (
        UnitConversionService,
    )

logger = logging.getLogger(__name__)


@dataclass
class UnitCategory:
    """Defines a unit category with its SI base unit and display options."""

    name: str  # Internal name (e.g., "temperature", "pressure")
    display_name: str  # Human-readable name (e.g., "Temperature", "Pressure")
    si_unit: str  # SI base unit (e.g., "K", "Pa")
    default_display_unit: str  # Default unit for display (e.g., "°C", "atm")
    available_units: list[str] = field(default_factory=list)
    description: str = ""


# Define all unit categories used in the application
UNIT_CATEGORIES: dict[str, UnitCategory] = {
    "temperature": UnitCategory(
        name="temperature",
        display_name="Temperature",
        si_unit="K",
        default_display_unit="°C",
        available_units=["°C", "°F", "K"],
        description="Temperature measurements",
    ),
    "pressure": UnitCategory(
        name="pressure",
        display_name="Pressure",
        si_unit="Pa",
        default_display_unit="atm",
        available_units=["atm", "bar", "psi", "kPa", "MPa", "Pa", "torr", "mmHg"],
        description="Pressure measurements",
    ),
    "mass_flow": UnitCategory(
        name="mass_flow",
        display_name="Mass Flow Rate",
        si_unit="kg/s",
        default_display_unit="kg/hr",
        available_units=["kg/hr", "kg/s", "lb/hr", "lb/s", "tonne/hr", "ton/hr", "g/s"],
        description="Mass flow rate measurements",
    ),
    "volumetric_flow": UnitCategory(
        name="volumetric_flow",
        display_name="Volumetric Flow Rate",
        si_unit="m3/s",
        default_display_unit="SCFM",
        available_units=[
            "SCFM",
            "m3/hr",
            "m3/s",
            "L/min",
            "L/s",
            "ft3/min",
            "gpm",
            "gal/hr",
        ],
        description="Volumetric flow rate measurements",
    ),
    "power": UnitCategory(
        name="power",
        display_name="Power / Heat Rate",
        si_unit="W",
        default_display_unit="kW",
        available_units=["kW", "MW", "W", "MMBTU/hr", "BTU/hr", "kcal/hr", "hp"],
        description="Power and heat rate measurements",
    ),
    "energy": UnitCategory(
        name="energy",
        display_name="Energy",
        si_unit="J",
        default_display_unit="kJ",
        available_units=["kJ", "MJ", "J", "BTU", "kcal", "kWh", "therm"],
        description="Energy measurements",
    ),
    "mass": UnitCategory(
        name="mass",
        display_name="Mass",
        si_unit="kg",
        default_display_unit="kg",
        available_units=["kg", "g", "lb", "tonne", "ton", "oz"],
        description="Mass measurements",
    ),
    "length": UnitCategory(
        name="length",
        display_name="Length",
        si_unit="m",
        default_display_unit="m",
        available_units=["m", "cm", "mm", "ft", "in", "km", "mi"],
        description="Length measurements",
    ),
    "volume": UnitCategory(
        name="volume",
        display_name="Volume",
        si_unit="m3",
        default_display_unit="L",
        available_units=["L", "m3", "mL", "gal", "ft3", "in3"],
        description="Volume measurements",
    ),
    "density": UnitCategory(
        name="density",
        display_name="Density",
        si_unit="kg/m3",
        default_display_unit="kg/m3",
        available_units=["kg/m3", "g/L", "g/cm3", "lb/ft3", "lb/gal"],
        description="Density measurements",
    ),
    "specific_heat": UnitCategory(
        name="specific_heat",
        display_name="Specific Heat",
        si_unit="J/kg·K",
        default_display_unit="kJ/kg·K",
        available_units=["kJ/kg·K", "J/kg·K", "BTU/lb·°F", "cal/g·°C"],
        description="Specific heat capacity",
    ),
    "heating_value": UnitCategory(
        name="heating_value",
        display_name="Heating Value",
        si_unit="MJ/kg",
        default_display_unit="MJ/kg",
        available_units=["MJ/kg", "kJ/kg", "BTU/lb", "kcal/kg", "MJ/Nm³", "BTU/scf"],
        description="Heating value (HHV/LHV)",
    ),
    "composition": UnitCategory(
        name="composition",
        display_name="Composition",
        si_unit="dimensionless",
        default_display_unit="vol %",
        available_units=[
            "vol %",
            "mass %",
            "mole fraction",
            "mass fraction",
            "ppm",
            "ppb",
        ],
        description="Gas/mixture composition",
    ),
    "area": UnitCategory(
        name="area",
        display_name="Area",
        si_unit="m2",
        default_display_unit="m2",
        available_units=["m2", "cm2", "mm2", "ft2", "in2"],
        description="Area measurements",
    ),
    "time": UnitCategory(
        name="time",
        display_name="Time",
        si_unit="s",
        default_display_unit="hr",
        available_units=["hr", "min", "s", "day"],
        description="Time measurements",
    ),
}

# Predefined unit system presets
UNIT_PRESETS: dict[str, dict[str, str]] = {
    "Default": {
        "temperature": "°C",
        "pressure": "atm",
        "mass_flow": "kg/hr",
        "volumetric_flow": "SCFM",
        "power": "kW",
        "energy": "kJ",
        "mass": "kg",
        "length": "m",
        "volume": "L",
        "density": "kg/m3",
        "specific_heat": "kJ/kg·K",
        "heating_value": "MJ/kg",
        "composition": "vol %",
        "area": "m2",
        "time": "hr",
    },
    "Metric (SI)": {
        "temperature": "K",
        "pressure": "Pa",
        "mass_flow": "kg/s",
        "volumetric_flow": "m3/s",
        "power": "W",
        "energy": "J",
        "mass": "kg",
        "length": "m",
        "volume": "m3",
        "density": "kg/m3",
        "specific_heat": "J/kg·K",
        "heating_value": "MJ/kg",
        "composition": "mole fraction",
        "area": "m2",
        "time": "s",
    },
    "Metric (Engineering)": {
        "temperature": "°C",
        "pressure": "bar",
        "mass_flow": "kg/hr",
        "volumetric_flow": "m3/hr",
        "power": "kW",
        "energy": "kJ",
        "mass": "kg",
        "length": "m",
        "volume": "L",
        "density": "kg/m3",
        "specific_heat": "kJ/kg·K",
        "heating_value": "MJ/kg",
        "composition": "vol %",
        "area": "m2",
        "time": "hr",
    },
    "Imperial (US)": {
        "temperature": "°F",
        "pressure": "psi",
        "mass_flow": "lb/hr",
        "volumetric_flow": "SCFM",
        "power": "BTU/hr",
        "energy": "BTU",
        "mass": "lb",
        "length": "ft",
        "volume": "gal",
        "density": "lb/ft3",
        "specific_heat": "BTU/lb·°F",
        "heating_value": "BTU/lb",
        "composition": "vol %",
        "area": "ft2",
        "time": "hr",
    },
}


class UnitPreferencesManager(QObject):
    """Manages global unit preferences for the application."""

    preferences_changed = pyqtSignal()
    category_unit_changed = pyqtSignal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.settings = QSettings("UpstreamDriftTools", "UnitPreferences")
        self._converter: UnitConversionService | None = None
        self._preferences: dict[str, str] = {}
        self._preset_name: str = "Default"
        self._callbacks: list[Callable[[], None]] = []

        self._load_preferences()

    @property
    def converter(self) -> UnitConversionService:
        """Lazily load the unit conversion service."""
        if self._converter is None:
            from ...calculators.conversion.service import get_service

            self._converter = get_service()
        return self._converter

    def _load_preferences(self) -> None:
        """Load unit preferences from settings."""
        saved_prefs = self.settings.value("unit_preferences", "{}")
        try:
            if isinstance(saved_prefs, str):
                self._preferences = json.loads(saved_prefs)
            else:
                self._preferences = {}
        except json.JSONDecodeError:
            self._preferences = {}

        self._preset_name = (
            str(self.settings.value("preset_name", "Default")) or "Default"
        )

        for cat_name, cat_info in UNIT_CATEGORIES.items():
            if cat_name not in self._preferences:
                self._preferences[cat_name] = cat_info.default_display_unit

    def _save_preferences(self) -> None:
        """Save unit preferences to settings."""
        self.settings.setValue("unit_preferences", json.dumps(self._preferences))
        self.settings.setValue("preset_name", self._preset_name)

    def get_preferred_unit(self, category: str) -> str:
        """Get the user's preferred unit for a category."""
        return self._preferences.get(
            category,
            UNIT_CATEGORIES.get(
                category, UnitCategory("", "", "", "")
            ).default_display_unit,
        )

    def set_preferred_unit(self, category: str, unit: str) -> None:
        """Set the user's preferred unit for a category."""
        if (
            category in UNIT_CATEGORIES
            and unit in UNIT_CATEGORIES[category].available_units
        ):
            old_unit = self._preferences.get(category)
            if old_unit != unit:
                self._preferences[category] = unit
                self._save_preferences()
                self.category_unit_changed.emit(category, unit)
                self.preferences_changed.emit()

    def get_si_unit(self, category: str) -> str:
        """Get the SI base unit for a category."""
        return UNIT_CATEGORIES[category].si_unit if category in UNIT_CATEGORIES else ""

    def convert_to_si(
        self, value: float, category: str, from_unit: str | None = None
    ) -> float:
        """Convert a value to SI units."""
        assert value is not None, "value must be provided"
        assert value is not None, "value must be provided"
        from_unit = from_unit or self.get_preferred_unit(category)
        si_unit = self.get_si_unit(category)
        if not from_unit or not si_unit or from_unit == si_unit:
            return value
        try:
            return self.converter.convert(value, from_unit, si_unit).value
        except (ValueError, KeyError, ZeroDivisionError):
            return value

    def convert_from_si(
        self, value: float, category: str, to_unit: str | None = None
    ) -> float:
        """Convert a value from SI units to display units."""
        assert value is not None, "value must be provided"
        assert value is not None, "value must be provided"
        to_unit = to_unit or self.get_preferred_unit(category)
        si_unit = self.get_si_unit(category)
        if not to_unit or not si_unit or to_unit == si_unit:
            return value
        try:
            return self.converter.convert(value, si_unit, to_unit).value
        except (ValueError, KeyError, ZeroDivisionError):
            return value


class _PreferencesHolder:
    """Singleton holder for UnitPreferencesManager (avoids global keyword)."""

    instance: UnitPreferencesManager | None = None


def get_unit_preferences_manager() -> UnitPreferencesManager:
    """Get the global UnitPreferencesManager instance."""
    if _PreferencesHolder.instance is None:
        _PreferencesHolder.instance = UnitPreferencesManager()
    return _PreferencesHolder.instance


__all__ = [
    "UNIT_CATEGORIES",
    "UNIT_PRESETS",
    "UnitCategory",
    "UnitPreferencesManager",
    "get_unit_preferences_manager",
]
