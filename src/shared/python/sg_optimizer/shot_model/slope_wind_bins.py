"""Slope/stance bins and wind geometry for the dispersion tables (SHOW-B6, #14570).

A sloping lie changes the strike before the ball flies: toes up/down tilts the
swing arc (attack angle), and ball above/below the feet flattens/steepens the
plane (face closes/opens, path moves in-to-out/out-to-in).  Each bin is a
deterministic shift of ``IMPACT_FIELDS`` rows, applied before the landing
simulator, like the B5 lie modifier (``physics.lie_interaction``).  Wind is a
steady vector resolved into the aim frame; the bins hold ``(headwind, crosswind)``
in mph.

**Every effect size is a labelled assumption** (``SLOPE_PARAMETERS``): the signs
follow standard golf-instruction practice (uphill adds loft, ball-above-feet
draws), the magnitudes are priors with no primary citation, and they are
club-independent.  Scope: stance and ball position only; slope along the ball
flight (roll, bounce) and firmness are not modelled here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.physics.lie_interaction import ParameterRecord
from src.shared.python.sg_optimizer.shot_model.skill_family import IMPACT_FIELDS

Array = NDArray[np.float64]
_ASSUMED = "Assumed prior; sign from golf instruction practice, magnitude unmeasured"
SLOPE_PARAMETERS: dict[str, ParameterRecord] = {
    "attack_deg_per_stance_deg": ParameterRecord(
        0.7,
        "assumption",
        _ASSUMED,
        "the arc follows most of the slope: uphill adds loft",
    ),
    "speed_loss_per_slope_deg": ParameterRecord(
        0.004, "assumption", _ASSUMED, "balance loss on any uneven stance"
    ),
    "face_deg_per_feet_deg": ParameterRecord(
        -0.4, "assumption", _ASSUMED, "ball above feet: flatter plane closes the face"
    ),
    "path_deg_per_feet_deg": ParameterRecord(
        0.3, "assumption", _ASSUMED, "ball above feet: path moves in-to-out"
    ),
}


@dataclass(frozen=True)
class SlopeBin:
    """Stance slope (toes-up positive) and ball-vs-feet slope (above positive), deg."""

    stance_deg: float = 0.0
    feet_deg: float = 0.0


SLOPE_BINS: dict[str, SlopeBin] = {
    "flat": SlopeBin(),
    "up_5": SlopeBin(stance_deg=5.0),
    "down_5": SlopeBin(stance_deg=-5.0),
    "above_5": SlopeBin(feet_deg=5.0),
    "below_5": SlopeBin(feet_deg=-5.0),
}
_COL = {name: i for i, name in enumerate(IMPACT_FIELDS)}


def make_slope_modifier(slope: str) -> Callable[[Array], Array]:
    """Deterministic ``IMPACT_FIELDS`` rows -> rows shift for slope bin ``slope``."""
    require(slope in SLOPE_BINS, f"unknown slope bin; have {tuple(SLOPE_BINS)}", slope)
    bin_ = SLOPE_BINS[slope]
    p = {k: v.value for k, v in SLOPE_PARAMETERS.items()}

    def modify(rows: Array) -> Array:
        require(rows.ndim == 2 and rows.shape[1] == len(IMPACT_FIELDS), "impact rows")
        out = rows.copy()
        total = abs(bin_.stance_deg) + abs(bin_.feet_deg)
        out[:, _COL["clubhead_speed_mph"]] *= (
            1.0 - p["speed_loss_per_slope_deg"] * total
        )
        out[:, _COL["attack_angle_deg"]] += (
            p["attack_deg_per_stance_deg"] * bin_.stance_deg
        )
        out[:, _COL["face_angle_deg"]] += p["face_deg_per_feet_deg"] * bin_.feet_deg
        out[:, _COL["club_path_deg"]] += p["path_deg_per_feet_deg"] * bin_.feet_deg
        ensure(out.shape == rows.shape, "row shape is preserved")
        return out

    return modify


def wind_to_aim_frame(speed_mph: float, from_bearing_deg: float) -> tuple[float, float]:
    """``(headwind, crosswind)`` mph of wind blowing *from* ``from_bearing_deg``.

    Bearing is measured from the target line: 0 is a pure headwind, 180 a pure
    tailwind (negative headwind), +90 a crosswind from the right (positive).
    """
    require(speed_mph >= 0.0, "wind speed must be >= 0", speed_mph)
    rad = math.radians(from_bearing_deg)
    return speed_mph * math.cos(rad), speed_mph * math.sin(rad)
