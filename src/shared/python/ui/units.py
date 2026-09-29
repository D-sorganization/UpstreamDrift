"""Unified display unit system and conversion helpers across tools (issue #8886).

All physical simulations and internal data models strictly store and compute in SI
units (m, m/s, kg, rad). This module provides display-layer formatting and conversions
according to user preferences (Metric vs Imperial).
"""

from __future__ import annotations

import enum
import math
from typing import Any

from src.shared.python.core.physics_constants import (
    FT_TO_M,
    KG_TO_LB,
    M_TO_FT,
    M_TO_YARD,
    MPH_TO_MPS,
    MPS_TO_MPH,
    YARD_TO_M,
)
from src.shared.python.data_io.user_config_root import user_config_dir

_LB_TO_KG = 1.0 / float(KG_TO_LB)
_FT_TO_M = float(FT_TO_M)
_YARD_TO_M = float(YARD_TO_M)
_MPH_TO_MPS = float(MPH_TO_MPS)


class UnitSystem(str, enum.Enum):
    """Supported display unit systems."""

    METRIC = "metric"
    IMPERIAL = "imperial"

    def __str__(self) -> str:
        return self.value


def get_unit_preference() -> UnitSystem:
    """Read the active unit preference from user configuration.

    Falls back to :attr:`UnitSystem.METRIC` if unconfigured.
    """
    try:
        from src.shared.python.ui.preferences_dialog import UserPreferences

        prefs = UserPreferences.load()
        raw = getattr(prefs, "unit_system", "metric")
        return UnitSystem(str(raw).lower())
    except Exception:
        return UnitSystem.METRIC


def set_unit_preference(system: UnitSystem | str) -> None:
    """Persist the user's unit system preference across settings stores."""
    if isinstance(system, str):
        try:
            unit_sys = UnitSystem(system.lower())
        except ValueError:
            raise ValueError(
                f"Invalid UnitSystem: {system!r}. Must be 'metric' or 'imperial'."
            ) from None
    elif isinstance(system, UnitSystem):
        unit_sys = system
    else:
        raise ValueError(
            f"Invalid UnitSystem: {system!r}. Must be 'metric' or 'imperial'."
        )

    # 1. Update preferences.json
    try:
        from src.shared.python.ui.preferences_dialog import UserPreferences

        prefs = UserPreferences.load()
        prefs.unit_system = unit_sys.value
        prefs.save()
    except Exception:
        pass


# ---- Distance conversions ----------------------------------------------------


def to_display_distance(
    val_m: float,
    system: UnitSystem | None = None,
    *,
    unit: str = "auto",
) -> float:
    """Convert distance in meters to the display unit system.

    Parameters
    ----------
    val_m:
        Distance in meters (SI).
    system:
        Target unit system. If None, uses active preference.
    unit:
        When system is IMPERIAL:
        - "yd" or "auto": convert to yards.
        - "ft": convert to feet.
    """
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        return val_m
    if unit == "ft":
        return val_m / _FT_TO_M
    return val_m / _YARD_TO_M


def from_display_distance(
    val_display: float,
    system: UnitSystem | None = None,
    *,
    unit: str = "auto",
) -> float:
    """Convert display distance to SI meters."""
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        return val_display
    if unit == "ft":
        return val_display * _FT_TO_M
    return val_display * _YARD_TO_M


def distance_suffix(
    system: UnitSystem | None = None,
    *,
    unit: str = "auto",
) -> str:
    """Return the display suffix for distance (e.g. ' m', ' yd', ' ft')."""
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        return " m"
    if unit == "ft":
        return " ft"
    return " yd"


def format_distance(
    val_m: float,
    system: UnitSystem | None = None,
    *,
    include_secondary: bool = True,
    unit: str = "yd",
    decimals: int = 1,
) -> str:
    """Format distance with primary unit and optional secondary parenthetical."""
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        primary = f"{val_m:.{decimals}f} m"
        if not include_secondary:
            return primary
        sec_val = val_m * (M_TO_FT if unit == "ft" else M_TO_YARD)
        sec_unit = "ft" if unit == "ft" else "yd"
        return f"{primary} ({sec_val:.{decimals}f} {sec_unit})"

    sec_unit = "ft" if unit == "ft" else "yd"
    disp_val = val_m * (M_TO_FT if unit == "ft" else M_TO_YARD)
    primary = f"{disp_val:.{decimals}f} {sec_unit}"
    if not include_secondary:
        return primary
    return f"{primary} ({val_m:.{decimals}f} m)"


# ---- Speed conversions -------------------------------------------------------


def to_display_speed(
    val_ms: float,
    system: UnitSystem | None = None,
) -> float:
    """Convert speed in m/s to the display unit system."""
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        return val_ms
    return val_ms / _MPH_TO_MPS


def from_display_speed(
    val_display: float,
    system: UnitSystem | None = None,
) -> float:
    """Convert display speed to SI m/s."""
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        return val_display
    return val_display * _MPH_TO_MPS


def speed_suffix(system: UnitSystem | None = None) -> str:
    """Return the display suffix for speed (' m/s' or ' mph')."""
    sys = system or get_unit_preference()
    return " m/s" if sys == UnitSystem.METRIC else " mph"


def format_speed(
    val_ms: float,
    system: UnitSystem | None = None,
    *,
    include_secondary: bool = True,
    decimals: int = 1,
) -> str:
    """Format speed with primary unit and optional secondary parenthetical."""
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        primary = f"{val_ms:.{decimals}f} m/s"
        if not include_secondary:
            return primary
        mph = val_ms * MPS_TO_MPH
        return f"{primary} ({mph:.{decimals}f} mph)"

    mph = val_ms * MPS_TO_MPH
    primary = f"{mph:.{decimals}f} mph"
    if not include_secondary:
        return primary
    return f"{primary} ({val_ms:.{decimals}f} m/s)"


# ---- Mass conversions --------------------------------------------------------


def to_display_mass(
    val_kg: float,
    system: UnitSystem | None = None,
) -> float:
    """Convert mass in kg to display unit system."""
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        return val_kg
    return val_kg * KG_TO_LB


def from_display_mass(
    val_display: float,
    system: UnitSystem | None = None,
) -> float:
    """Convert display mass to SI kg."""
    sys = system or get_unit_preference()
    if sys == UnitSystem.METRIC:
        return val_display
    return val_display * _LB_TO_KG


def mass_suffix(system: UnitSystem | None = None) -> str:
    """Return the display suffix for mass (' kg' or ' lb')."""
    sys = system or get_unit_preference()
    return " kg" if sys == UnitSystem.METRIC else " lb"


# ---- Spin formatting ---------------------------------------------------------


def spin_suffix() -> str:
    """Return spin rate suffix (' rpm')."""
    return " rpm"


def format_spin(backspin_rpm: float, sidespin_rpm: float | None = None) -> str:
    """Format spin rate."""
    if sidespin_rpm is None:
        return f"{backspin_rpm:.0f} rpm"
    return f"{backspin_rpm:.0f} rpm backspin, {sidespin_rpm:.0f} rpm sidespin"


__all__ = [
    "UnitSystem",
    "get_unit_preference",
    "set_unit_preference",
    "to_display_distance",
    "from_display_distance",
    "distance_suffix",
    "format_distance",
    "to_display_speed",
    "from_display_speed",
    "speed_suffix",
    "format_speed",
    "to_display_mass",
    "from_display_mass",
    "mass_suffix",
    "spin_suffix",
    "format_spin",
]
