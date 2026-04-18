"""
Comprehensive unit conversion system for the Pendulum Simulator.

Modeled after the Gasification Model's dropdown-based unit selection.
Supports per-category unit selection with presets and persistence.

Design by Contract
------------------
- UnitConverter is stateless; all methods are pure functions.
- UnitPreferences stores user's per-category selections.
- All conversion factors are exact NIST values.

DRY
---
Single source of truth for all conversion constants and unit labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.shared.python.pendulum_simulator.constants import GRAVITY_STANDARD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (exact NIST values)
# ---------------------------------------------------------------------------
_METERS_PER_INCH = 0.0254
_METERS_PER_FOOT = 0.3048
_KG_PER_LB = 0.45359237
_NM_PER_LBFIN = 0.1129848290276167
_NM_PER_LBFFT = 1.3558179483314
_NM_PER_KGFM = GRAVITY_STANDARD
_N_PER_LBF = 4.4482216152605
_J_PER_FTLBF = 1.3558179483314
_W_PER_FTLBFS = 1.3558179483314
_RAD_PER_DEG = 0.017453292519943295


# ---------------------------------------------------------------------------
# Unit category definitions
# ---------------------------------------------------------------------------


class UnitCategory(Enum):
    """Categories of physical quantities."""

    LENGTH = "length"
    MASS = "mass"
    TORQUE = "torque"
    FORCE = "force"
    ANGLE = "angle"
    ANGULAR_VELOCITY = "angular_velocity"
    STIFFNESS = "stiffness"
    ENERGY = "energy"
    POWER = "power"
    LINEAR_VELOCITY = "linear_velocity"


# Each unit: (label, factor_to_si)
# factor_to_si: multiply display value by this to get SI
_UNIT_OPTIONS: dict[UnitCategory, list[tuple[str, float]]] = {
    UnitCategory.LENGTH: [
        ("m", 1.0),
        ("cm", 0.01),
        ("in", _METERS_PER_INCH),
        ("ft", _METERS_PER_FOOT),
    ],
    UnitCategory.MASS: [
        ("kg", 1.0),
        ("g", 0.001),
        ("lb", _KG_PER_LB),
    ],
    UnitCategory.TORQUE: [
        ("NÂ·m", 1.0),
        ("lbfÂ·in", _NM_PER_LBFIN),
        ("lbfÂ·ft", _NM_PER_LBFFT),
        ("kgfÂ·m", _NM_PER_KGFM),
    ],
    UnitCategory.FORCE: [
        ("N", 1.0),
        ("lbf", _N_PER_LBF),
        ("kgf", GRAVITY_STANDARD),
    ],
    UnitCategory.ANGLE: [
        ("rad", 1.0),
        ("deg", _RAD_PER_DEG),
    ],
    UnitCategory.ANGULAR_VELOCITY: [
        ("rad/s", 1.0),
        ("deg/s", _RAD_PER_DEG),
        ("rpm", 0.10471975511965978),  # 2Ï€/60
    ],
    UnitCategory.STIFFNESS: [
        ("NÂ·m/rad", 1.0),
        ("lbfÂ·in/rad", _NM_PER_LBFIN),
        ("lbfÂ·ft/rad", _NM_PER_LBFFT),
    ],
    UnitCategory.ENERGY: [
        ("J", 1.0),
        ("ftÂ·lbf", _J_PER_FTLBF),
        ("cal", 4.184),
    ],
    UnitCategory.POWER: [
        ("W", 1.0),
        ("ftÂ·lbf/s", _W_PER_FTLBFS),
        ("hp", 745.69987158227022),
    ],
    UnitCategory.LINEAR_VELOCITY: [
        ("m/s", 1.0),
        ("ft/s", _METERS_PER_FOOT),
        ("mph", 0.44704),
    ],
}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


_PRESETS: dict[str, dict[UnitCategory, str]] = {
    "SI": {cat: opts[0][0] for cat, opts in _UNIT_OPTIONS.items()},
    "Imperial": {
        UnitCategory.LENGTH: "in",
        UnitCategory.MASS: "lb",
        UnitCategory.TORQUE: "lbfÂ·in",
        UnitCategory.FORCE: "lbf",
        UnitCategory.ANGLE: "deg",
        UnitCategory.ANGULAR_VELOCITY: "deg/s",
        UnitCategory.STIFFNESS: "lbfÂ·in/rad",
        UnitCategory.ENERGY: "ftÂ·lbf",
        UnitCategory.POWER: "ftÂ·lbf/s",
        UnitCategory.LINEAR_VELOCITY: "ft/s",
    },
    "Engineering": {
        UnitCategory.LENGTH: "cm",
        UnitCategory.MASS: "kg",
        UnitCategory.TORQUE: "NÂ·m",
        UnitCategory.FORCE: "N",
        UnitCategory.ANGLE: "deg",
        UnitCategory.ANGULAR_VELOCITY: "deg/s",
        UnitCategory.STIFFNESS: "NÂ·m/rad",
        UnitCategory.ENERGY: "J",
        UnitCategory.POWER: "W",
        UnitCategory.LINEAR_VELOCITY: "m/s",
    },
}


# ---------------------------------------------------------------------------
# Unit preferences
# ---------------------------------------------------------------------------


@dataclass
class UnitPreferences:
    """Stores the user's per-category unit selections.

    Contract:
        - All selected units must be valid options for their category.
        - apply_preset() sets all categories to the preset's values.
    """

    selections: dict[UnitCategory, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Default to SI if empty
        if not self.selections:
            self.apply_preset("SI")

    def apply_preset(self, preset_name: str) -> None:
        """Apply a named preset.

        Pre: preset_name in _PRESETS.
        """
        assert preset_name in _PRESETS, f"Unknown preset: {preset_name}"
        self.selections = dict(_PRESETS[preset_name])
        logger.info("Applied unit preset: %s", preset_name)

    def get_unit(self, category: UnitCategory) -> str:
        """Get the current unit label for a category."""
        return self.selections.get(category, _UNIT_OPTIONS[category][0][0])

    def set_unit(self, category: UnitCategory, unit_label: str) -> None:
        """Set the unit for a category.

        Pre: unit_label must be a valid option for the category.
        """
        valid = [label for label, _ in _UNIT_OPTIONS[category]]
        assert (
            unit_label in valid
        ), f"Invalid unit '{unit_label}' for {category.value}. Valid: {valid}"
        self.selections[category] = unit_label

    def save_to_qsettings(self, settings: Any) -> None:
        """Persist selections to QSettings."""
        for cat, unit in self.selections.items():
            settings.setValue(f"units/{cat.value}", unit)
        logger.debug("Saved unit preferences to QSettings")

    def load_from_qsettings(self, settings: Any) -> None:
        """Restore selections from QSettings."""
        for cat in UnitCategory:
            val = settings.value(f"units/{cat.value}")
            if val is not None and isinstance(val, str):
                try:
                    self.set_unit(cat, val)
                except AssertionError:
                    logger.warning(
                        "Ignoring invalid saved unit '%s' for %s",
                        val,
                        cat.value,
                    )


# ---------------------------------------------------------------------------
# Stateless conversion functions
# ---------------------------------------------------------------------------


def _get_factor(category: UnitCategory, unit_label: str) -> float:
    """Get the to-SI conversion factor for a unit.

    Pre: unit_label is valid for category.
    Post: returns positive float.
    """
    for label, factor in _UNIT_OPTIONS[category]:
        if label == unit_label:
            assert factor > 0
            return factor
    raise KeyError(f"Unknown unit '{unit_label}' for {category.value}")


def to_si(value: float, category: UnitCategory, prefs: UnitPreferences) -> float:
    """Convert a display value to SI units.

    Pre: value is finite.
    Post: result is in SI.
    """
    assert value is not None, "value must be provided"
    unit = prefs.get_unit(category)
    factor = _get_factor(category, unit)
    return value * factor


def from_si(value: float, category: UnitCategory, prefs: UnitPreferences) -> float:
    """Convert an SI value to display units.

    Pre: value is finite.
    Post: result is in current display units.
    """
    assert value is not None, "value must be provided"
    unit = prefs.get_unit(category)
    factor = _get_factor(category, unit)
    return value / factor


def get_unit_label(category: UnitCategory, prefs: UnitPreferences) -> str:
    """Get the current display unit label."""
    return prefs.get_unit(category)


def get_available_units(category: UnitCategory) -> list[str]:
    """List all available units for a category."""
    return [label for label, _ in _UNIT_OPTIONS[category]]


def get_preset_names() -> list[str]:
    """List all available preset names."""
    return list(_PRESETS.keys())


# ---------------------------------------------------------------------------
# Legacy compatibility (drop-in replacement for old UnitConverter)
# ---------------------------------------------------------------------------


class UnitSystem(Enum):
    """Supported unit systems (legacy API)."""

    SI = "SI"
    IMPERIAL = "Imperial"


@dataclass(frozen=True)
class UnitConverter:
    """Legacy bidirectional converter (thin wrapper over new system).

    Maintained for backward compatibility with existing GUI code.
    """

    system: UnitSystem = UnitSystem.SI

    def _prefs(self) -> UnitPreferences:
        prefs = UnitPreferences()
        if self.system == UnitSystem.IMPERIAL:
            prefs.apply_preset("Imperial")
        return prefs

    def to_si_length(self, value: float) -> float:
        return to_si(value, UnitCategory.LENGTH, self._prefs())

    def from_si_length(self, meters: float) -> float:
        return from_si(meters, UnitCategory.LENGTH, self._prefs())

    def to_si_mass(self, value: float) -> float:
        return to_si(value, UnitCategory.MASS, self._prefs())

    def from_si_mass(self, kg: float) -> float:
        return from_si(kg, UnitCategory.MASS, self._prefs())

    def to_si_torque(self, value: float) -> float:
        return to_si(value, UnitCategory.TORQUE, self._prefs())

    def from_si_torque(self, nm: float) -> float:
        return from_si(nm, UnitCategory.TORQUE, self._prefs())

    @property
    def length_unit(self) -> str:
        return get_unit_label(UnitCategory.LENGTH, self._prefs())

    @property
    def mass_unit(self) -> str:
        return get_unit_label(UnitCategory.MASS, self._prefs())

    @property
    def torque_unit(self) -> str:
        return get_unit_label(UnitCategory.TORQUE, self._prefs())

    @property
    def velocity_unit(self) -> str:
        return "rad/s"

    @property
    def angle_unit(self) -> str:
        return "Â°"
