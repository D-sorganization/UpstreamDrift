"""Display units of the native exporter's distance and speed labels (#15443).

Distances are drawn in yards by default (golf convention) and speeds in mph;
both are settings of ``display_units_settings.json`` (a command line option is
a later slice). Only label text changes: internal numbers, the flight model
and manifest values stay SI. The yard factor is the repository's
``core.physics_constants.YARD_TO_M`` (exactly 0.9144 m); small lengths (mm, the
low point in m) are not converted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.core.physics_constants import YARD_TO_M
from src.shared.python.impact_parameters.extract import MPS_TO_MPH

SETTINGS_PATH = Path(__file__).with_name("display_units_settings.json")
DISTANCE_UNITS = ("yd", "m")
SPEED_UNITS = ("mph", "m/s")
#: metres per yard, exactly (1959 international yard and pound agreement)
METRES_PER_YARD = float(YARD_TO_M)


@dataclass(frozen=True)
class DisplayUnits:
    """Units of the distance and speed labels.

    Raises ``ValueError`` for a unit outside ``DISTANCE_UNITS``/``SPEED_UNITS``
    and for negative or non-integer ``distance_decimals``.
    """

    distance_unit: str = "yd"
    speed_unit: str = "mph"
    distance_decimals: int = 0

    def __post_init__(self) -> None:
        if self.distance_unit not in DISTANCE_UNITS:
            raise ValueError(f"distance_unit must be one of {DISTANCE_UNITS}")
        if self.speed_unit not in SPEED_UNITS:
            raise ValueError(f"speed_unit must be one of {SPEED_UNITS}")
        decimals = self.distance_decimals
        if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
            raise ValueError("distance_decimals must be a non-negative integer")

    @property
    def distance_scale(self) -> float:
        """Display units per metre."""
        return 1.0 / METRES_PER_YARD if self.distance_unit == "yd" else 1.0


METRIC = DisplayUnits("m", "mph")


def load_display_units(path: Path | None = None) -> DisplayUnits:
    """Units from ``path`` (default: the file beside this module).

    Raises ``ValueError``/``TypeError`` for an unknown unit or entry.
    """
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    try:
        return DisplayUnits(**document)
    except TypeError as exc:
        raise ValueError(f"display units settings: {exc}") from exc


def convert_distance(metres: float, units: DisplayUnits) -> float:
    """``metres`` in the display distance unit. Raises ``ValueError`` if not finite."""
    if not math.isfinite(metres):
        raise ValueError(f"metres must be finite, got {metres}")
    return metres * units.distance_scale


def distance_text(metres: float, units: DisplayUnits) -> str:
    """Label text of a distance, e.g. ``"226 yd"``; the sign is kept."""
    value = convert_distance(metres, units)
    return f"{value:.{units.distance_decimals}f} {units.distance_unit}"


def speed_text(mph: float, units: DisplayUnits) -> str:
    """Label text of a speed given in mph, e.g. ``"170 mph"`` or ``"76 m/s"``."""
    if not math.isfinite(mph):
        raise ValueError(f"mph must be finite, got {mph}")
    if units.speed_unit == "mph":
        return f"{mph:.0f} mph"
    return f"{mph / MPS_TO_MPH:.0f} m/s"
