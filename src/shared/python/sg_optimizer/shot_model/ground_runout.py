"""Bounce and roll after carry, for the SHOW-B3b dispersion tables (#14610).

The flight pipeline ends at the first ground contact (carry).  This adds the
rest position: a landing state ``(x, y, vx, vy, vz)`` (yards, then m/s in the
aim frame: x downrange, y left, z up; ``vz < 0`` descending) becomes resting
offsets.  Ground model ``sg_runout_v1``, per shot:

* **bounce** = ``bounce_gain * h * restitution * sin(descent)``, one hop along the
  horizontal landing direction (no spin decay);
* **roll** = ``roll_gain * h * cos(descent) * lie_roll * slope_roll``, same
  direction, where ``h`` is the horizontal landing speed (m/s), a steeper descent
  rolls less, ``lie_roll`` is a per-lie multiplier, and ``slope_roll =
  max(0.05, 1 - slope_roll_per_tan * tan(stance))`` (uphill shortens the roll,
  downhill lengthens it).

Slope uses the one B6 vocabulary (``slope_wind_bins.SLOPE_BINS``): the stance
grade is read as the grade of the landing zone, i.e. the landing zone is assumed
to share the lie's grade; ball-above/below-feet bins have no runout effect.  Every
coefficient is a labelled assumption (``GROUND_PARAMETERS``), fairway-first-contact
turf, not green physics.  Credit: first drafted in Cursor PR #14642.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.physics.lie_interaction import ParameterRecord
from src.shared.python.sg_optimizer.shot_model.slope_wind_bins import SLOPE_BINS

GROUND_MODEL_ID = "sg_runout_v1"
LANDING_COLUMNS = ("x_yd", "y_yd", "vx_ms", "vy_ms", "vz_ms")
_ASSUMED = "Assumed prior; unmeasured turf, no primary citation"


def _p(value: float, why: str) -> ParameterRecord:
    return ParameterRecord(value, "assumption", _ASSUMED, why)


GROUND_PARAMETERS: dict[str, ParameterRecord] = {
    "bounce_gain_yd_per_ms": _p(0.12, "yards of hop per m/s of horizontal speed"),
    "restitution": _p(0.45, "fraction of the hop kept on one turf bounce"),
    "roll_gain_yd_per_ms": _p(1.35, "yards of roll per m/s on flat fairway"),
    "slope_roll_per_tan": _p(1.8, "roll lost per unit of uphill grade (tan)"),
    "roll.tee": _p(1.0, "firm, short grass"),
    "roll.fairway": _p(1.0, "reference lie"),
    "roll.rough": _p(0.35, "long grass stops the ball"),
    "roll.sand": _p(0.15, "sand stops the ball"),
}
_MIN_SLOPE_MULT = 0.05


@dataclass(frozen=True)
class SlopeAwareGround:
    """``GroundModel``: landing states ``(n, 5)`` -> resting offsets ``(n, 2)`` yards."""

    model_id: str = GROUND_MODEL_ID

    def __call__(
        self, landing: NDArray[np.float64], lie: str, slope: str
    ) -> NDArray[np.float64]:
        require(
            landing.ndim == 2 and landing.shape[1] == len(LANDING_COLUMNS),
            f"landing state needs columns {LANDING_COLUMNS}",
            landing.shape,
        )
        require(slope in SLOPE_BINS, f"unknown slope bin; have {tuple(SLOPE_BINS)}")
        require(f"roll.{lie}" in GROUND_PARAMETERS, "unknown lie for runout", lie)
        p = {k: v.value for k, v in GROUND_PARAMETERS.items()}
        vx, vy, vz = landing[:, 2], landing[:, 3], landing[:, 4]
        h = np.hypot(vx, vy)
        descent = np.arctan2(-vz, h).clip(np.radians(5.0), np.radians(85.0))
        grade = np.tan(np.radians(SLOPE_BINS[slope].stance_deg))
        slope_mult = max(_MIN_SLOPE_MULT, 1.0 - p["slope_roll_per_tan"] * grade)
        bounce = p["bounce_gain_yd_per_ms"] * h * p["restitution"] * np.sin(descent)
        roll = p["roll_gain_yd_per_ms"] * h * np.cos(descent) * p[f"roll.{lie}"]
        run = bounce + roll * slope_mult
        unit = np.divide(
            np.column_stack([vx, vy]),
            h[:, None],
            out=np.zeros((len(h), 2)),
            where=h[:, None] > 0,
        )
        out = landing[:, :2] + run[:, None] * unit
        ensure(bool(np.isfinite(out).all()), "rest position must be finite")
        return out
