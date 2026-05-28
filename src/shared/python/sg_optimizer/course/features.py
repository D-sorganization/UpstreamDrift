"""StateFeatures dataclass — per-state feature vector for the strategy MDP.

Bridges the raster-based MDP state (integer lie code + ball position) with
continuous environmental features needed for condition-aware shot models.

Phase 2 (#6271).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LIE_NAMES

if TYPE_CHECKING:  # pragma: no cover
    from src.shared.python.sg_optimizer.course.rasterize import LieRaster
    from src.shared.python.sg_optimizer.mdp.state import State


# Lie strings for StateFeatures — a stable public type alias.
LieStr = Literal["fairway", "rough", "bunker", "green", "tee", "ob", "trees", "water"]

# Mapping from lie string to LIE_CODES value — we map "sand" → "bunker" for
# the StateFeatures API so the external surface uses the term golfers use.
_LIE_REMAP: dict[str, str] = {"sand": "bunker"}


def _lie_int_to_str(lie_code: int) -> LieStr:
    raw = LIE_NAMES.get(lie_code, "rough")
    remapped = _LIE_REMAP.get(raw, raw)
    return remapped  # type: ignore[return-value]


@dataclass(frozen=True)
class StateFeatures:
    """Continuous feature vector derived from an MDP state + course map.

    Fields
    ------
    distance_to_pin_m : float
        Straight-line distance to the pin in metres.
    distance_to_center_m : float
        Straight-line distance to the green centre in metres.  Equals
        distance_to_pin_m when pin == green_center.
    lie : LieStr
        Human-readable lie class (uses "bunker" instead of "sand" to match
        golf convention).
    slope_deg : float
        Ground slope angle in degrees (positive = uphill toward the target).
        Default 0.0 for flat rasters.
    stimp : float
        Green speed in stimpmeter units (only meaningful when lie == "green").
    wind_mph : float
        Wind speed in miles per hour.
    wind_dir_deg : float
        Wind direction in degrees clockwise from North (meteorological convention).
    """

    distance_to_pin_m: float
    distance_to_center_m: float
    lie: LieStr
    slope_deg: float = 0.0
    stimp: float = 10.5
    wind_mph: float = 0.0
    wind_dir_deg: float = 0.0

    def __post_init__(self) -> None:
        require(
            self.distance_to_pin_m >= 0.0,
            "distance_to_pin_m must be >= 0",
            self.distance_to_pin_m,
        )
        require(
            self.distance_to_center_m >= 0.0,
            "distance_to_center_m must be >= 0",
            self.distance_to_center_m,
        )
        require(
            -90.0 <= self.slope_deg <= 90.0,
            "slope_deg must be in [-90, 90]",
            self.slope_deg,
        )
        require(8.0 <= self.stimp <= 14.0, "stimp must be in [8, 14]", self.stimp)
        require(self.wind_mph >= 0.0, "wind_mph must be >= 0", self.wind_mph)
        require(
            0.0 <= self.wind_dir_deg < 360.0,
            "wind_dir_deg must be in [0, 360)",
            self.wind_dir_deg,
        )

    # --- Factory method --------------------------------------------------

    @classmethod
    def from_state_and_course(
        cls,
        state: State,
        raster: LieRaster,
        *,
        stimp: float = 10.5,
        wind_mph: float = 0.0,
        wind_dir_deg: float = 0.0,
        slope_deg: float = 0.0,
        green_center: tuple[float, float] | None = None,
    ) -> StateFeatures:
        """Compute StateFeatures from an MDP state and a LieRaster.

        Parameters
        ----------
        state :
            Current ball position + lie code.
        raster :
            The rasterized hole map, used to get the pin position.
        stimp :
            Green speed; defaults to medium (10.5).
        wind_mph, wind_dir_deg :
            Wind conditions.
        slope_deg :
            Ground slope; not available from a flat raster — caller supplies.
        green_center :
            (x, y) of the green centre in hole-frame yards.  Falls back to
            ``raster.pin`` when not supplied.
        """
        require(
            state.lie in LIE_CODES.values(),
            f"invalid lie code {state.lie}",
            state.lie,
        )
        pin_x, pin_y = raster.pin
        dist_to_pin_m = (
            math.hypot(state.x - pin_x, state.y - pin_y) * 0.9144  # yards → metres
        )

        if green_center is not None:
            cx, cy = green_center
        else:
            cx, cy = raster.pin
        dist_to_center_m = math.hypot(state.x - cx, state.y - cy) * 0.9144

        lie_str = _lie_int_to_str(state.lie)

        return cls(
            distance_to_pin_m=dist_to_pin_m,
            distance_to_center_m=dist_to_center_m,
            lie=lie_str,
            slope_deg=slope_deg,
            stimp=stimp,
            wind_mph=wind_mph,
            wind_dir_deg=wind_dir_deg,
        )
