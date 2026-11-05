"""Post-carry bounce and roll for SHOW-B3b dispersion tables (#14610).

Ground model (aim frame: *x* downrange, *y* left, yards)
-------------------------------------------------------
1. **Carry** is the aerodynamic first-ground point (from the flight pipeline).
2. **Bounce** adds a short hop along the horizontal landing velocity; magnitude
   scales with ``restitution`` and descent angle (ASSUMPTION: one bounce, no
   spin decay model).
3. **Roll** continues along the landing direction; distance scales with
   horizontal landing speed, ``roll_gain``, lie rolling resistance, and slope
   grade toward +x (uphill shortens downrange roll, downhill extends it).

Every coefficient is an explicit assumption until a turf calibration lands;
see ``GROUND_MODEL_ID`` in table provenance.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require

M_TO_YD = 1.0 / 0.9144

# Slope bin -> grade toward +x (degrees), matching ``course.features.slope_deg``.
SLOPE_BINS: dict[str, float] = {
    "flat": 0.0,
    "uphill": 4.0,
    "downhill": -4.0,
}
SLOPES: tuple[str, ...] = tuple(SLOPE_BINS)

GROUND_MODEL_ID = "sg_runout_v1"
# ASSUMPTION: fairway-first-contact, not green Stimpmeter physics.
RESTITUTION = 0.45
ROLL_GAIN_YD_PER_MS = 1.35
BOUNCE_GAIN_YD_PER_MS = 0.12

_LIE_ROLL: dict[str, float] = {
    "tee": 1.0,
    "fairway": 1.0,
    "first_cut": 0.85,
    "rough": 0.35,
    "sand": 0.15,
    "bunker": 0.15,
    "hardpan": 1.1,
}

SKILL_CLUB: dict[str, str] = {
    "driver": "driver",
    "fairway_wood": "3_wood",
    "long_iron": "4_iron",
    "mid_iron": "7_iron",
    "short_iron": "9_iron",
    "wedge": "pw",
    "pitch": "gw",
    "chip": "sw",
    "bunker": "sw",
}


def _slope_along_mult(slope: str) -> float:
    require(slope in SLOPE_BINS, f"unknown slope bin {slope!r}")
    grade = math.radians(SLOPE_BINS[slope])
    # Uphill (+grade) fights roll; downhill assists.
    return max(0.05, 1.0 - 1.8 * math.tan(grade))


def _lie_roll_mult(lie: str) -> float:
    require(lie in _LIE_ROLL, f"unknown lie for roll {lie!r}")
    return _LIE_ROLL[lie]


def rest_position(
    carry_yds: NDArray[np.float64],
    *,
    slope: str,
    lie: str,
    landing_speed_ms: NDArray[np.float64] | None = None,
    descent_deg: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Carry offsets (n, 2) yards -> resting offsets on the tilted fairway plane."""
    require(carry_yds.ndim == 2 and carry_yds.shape[1] == 2, "carry_yds shape")
    n = len(carry_yds)
    if landing_speed_ms is None:
        landing_speed_ms = np.clip(carry_yds[:, 0] * 0.22, 8.0, 55.0)
    if descent_deg is None:
        descent_deg = np.full(n, 42.0)
    require(len(landing_speed_ms) == n and len(descent_deg) == n, "batch length")

    slope_m = _slope_along_mult(slope)
    lie_m = _lie_roll_mult(lie)
    descent = np.radians(np.clip(descent_deg, 5.0, 85.0))
    horiz = landing_speed_ms * M_TO_YD

    bounce = BOUNCE_GAIN_YD_PER_MS * horiz * RESTITUTION * np.sin(descent)
    roll = ROLL_GAIN_YD_PER_MS * horiz * lie_m * slope_m
    extra = bounce + roll

    out = carry_yds.copy()
    out[:, 0] += extra
    return out
