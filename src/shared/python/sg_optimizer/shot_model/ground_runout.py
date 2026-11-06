"""Bounce and roll after carry, for the SHOW-B3b dispersion tables (#14610).

The flight pipeline ends at the first ground contact (carry).  This adds the
rest position: a landing state ``(x, y, vx, vy, vz)`` (yards, then m/s in the
aim frame: x downrange, y left, z up; ``vz < 0`` descending) becomes resting
offsets.  Ground model ``sg_runout_v1``, per shot:

* **bounce** = ``bounce_yd_per_ms * h * sin(descent)``, one hop along the horizontal
  landing direction (no spin decay; the gain folds in the turf restitution, as only
  the product is identifiable);
* **roll** = ``roll_gain * h * cos(descent) * skill_roll * lie_roll * slope_roll``,
  same direction, where ``h`` is the horizontal landing speed (m/s), a steeper
  descent rolls less, ``skill_roll`` is a per-skill multiplier standing in for the
  spin and check the pipeline does not carry (``IMPACT_FIELDS`` has no spin column;
  it falls from driver to wedge so the roll fraction of carry does too),
  ``lie_roll`` is a per-lie multiplier, and ``slope_roll =
  max(0.05, 1 - slope_roll_per_tan * tan(stance))`` (uphill shortens the roll,
  downhill lengthens it).

Slope uses the one B6 vocabulary (``slope_wind_bins.SLOPE_BINS``): the stance
grade is read as the grade of the landing zone, i.e. the landing zone is assumed
to share the lie's grade, and likewise the landing surface (``lie_roll``) is the
impact lie's surface (a drive hit from the fairway is not allowed to finish in the
rough in these tables); ball-above/below-feet bins have no runout effect.  Every
coefficient is a labelled assumption (``GROUND_PARAMETERS``), fairway-first-contact
turf, not green physics.  Green-side skills (pitch, chip, bunker) keep carry: they land outside this fairway-turf
domain (``CARRY_ONLY_SKILLS``, hashed).  Credit: first drafted in Cursor PR #14642.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    "bounce_yd_per_ms": _p(
        0.054, "yards of hop per m/s (gain 0.12 x restitution 0.45)"
    ),
    "roll_gain_yd_per_ms": _p(1.35, "yards of roll per m/s on flat fairway"),
    "slope_roll_per_tan": _p(1.8, "roll lost per unit of uphill grade (tan)"),
    "skill.driver": _p(1.0, "low spin: releases most"),
    "skill.fairway_wood": _p(0.75, "a little more spin"),
    "skill.long_iron": _p(0.58, "iron spin begins to check the ball"),
    "skill.mid_iron": _p(0.45, "spin checks the ball"),
    "skill.short_iron": _p(0.3, "high spin"),
    "skill.wedge": _p(0.18, "high spin, steep descent"),
    "roll.tee": _p(1.0, "firm, short grass"),
    "roll.fairway": _p(1.0, "reference lie"),
    "roll.rough": _p(0.35, "long grass stops the ball"),
    "roll.sand": _p(0.15, "sand stops the ball"),
}
_MIN_SLOPE_MULT = 0.05
#: Green-side skills land on the green or in sand, outside this fairway-turf model:
#: they keep carry in rest mode (listed in the hashed provenance) until a green model.
CARRY_ONLY_SKILLS: tuple[str, ...] = ("pitch", "chip", "bunker")

#: Skill family key -> representative bag club for dispersion table labels (B3/B3b).
SKILL_CLUB: dict[str, str] = {
    "driver": "driver",
    "fairway_wood": "3_wood",
    "long_iron": "4_iron",
    "mid_iron": "6_iron",
    "short_iron": "9_iron",
    "wedge": "pw",
    "pitch": "gw",
    "chip": "sw",
    "bunker": "sw",
}


def rest_position(
    carry_yds: NDArray[np.float64],
    *,
    slope: str,
    lie: str,
    landing_speed_ms: NDArray[np.float64] | None = None,
    descent_deg: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Analytic carry (``n, 2`` yd) -> resting offsets via :class:`SlopeAwareGround`.

    Used by unit tests and simple stand-in simulators; the flight pipeline uses
    :class:`~src.shared.python.sg_optimizer.shot_model.physics_simulator.PipelineLandingSimulator`.
    """
    require(carry_yds.ndim == 2 and carry_yds.shape[1] == 2, "carry_yds shape", carry_yds.shape)
    n = len(carry_yds)
    if landing_speed_ms is None:
        landing_speed_ms = np.full(n, 35.0)
    if descent_deg is None:
        descent_deg = np.full(n, 42.0)
    require(len(landing_speed_ms) == n and len(descent_deg) == n, "batch length")
    vx = np.asarray(landing_speed_ms, dtype=np.float64)
    vz = -np.abs(
        np.tan(np.radians(np.asarray(descent_deg, dtype=np.float64))) * vx
    )
    landing = np.column_stack([carry_yds[:, 0], carry_yds[:, 1], vx, np.zeros(n), vz])
    return SlopeAwareGround()(landing, lie, slope, "mid_iron")


@dataclass(frozen=True)
class SlopeAwareGround:
    """``GroundModel``: landing states ``(n, 5)`` -> resting offsets ``(n, 2)`` yards."""

    model_id: str = GROUND_MODEL_ID

    @property
    def provenance(self) -> dict[str, Any]:
        """Hashed into the table: the id, every coefficient and the slope reading."""
        return {
            "ground_model": self.model_id,
            "ground_parameters": {k: v.value for k, v in GROUND_PARAMETERS.items()},
            "ground_carry_only_skills": list(CARRY_ONLY_SKILLS),
            "ground_slope_source": "stance bin (landing grade = lie grade)",
            "ground_lie_source": "impact lie bin (landing surface = impact lie)",
        }

    def __call__(
        self, landing: NDArray[np.float64], lie: str, slope: str, skill: str
    ) -> NDArray[np.float64]:
        require(
            landing.ndim == 2 and landing.shape[1] == len(LANDING_COLUMNS),
            f"landing state needs columns {LANDING_COLUMNS}",
            landing.shape,
        )
        require(slope in SLOPE_BINS, f"unknown slope bin; have {tuple(SLOPE_BINS)}")
        require(f"roll.{lie}" in GROUND_PARAMETERS, "unknown lie for runout", lie)
        known = f"skill.{skill}" in GROUND_PARAMETERS
        require(known or skill in CARRY_ONLY_SKILLS, "unknown skill for runout", skill)
        if skill in CARRY_ONLY_SKILLS:
            return np.array(landing[:, :2], dtype=np.float64)
        p = {k: v.value for k, v in GROUND_PARAMETERS.items()}
        vx, vy, vz = landing[:, 2], landing[:, 3], landing[:, 4]
        h = np.hypot(vx, vy)
        descent = np.arctan2(-vz, h).clip(np.radians(5.0), np.radians(85.0))
        grade = np.tan(np.radians(SLOPE_BINS[slope].stance_deg))
        slope_mult = max(_MIN_SLOPE_MULT, 1.0 - p["slope_roll_per_tan"] * grade)
        bounce = p["bounce_yd_per_ms"] * h * np.sin(descent)
        roll = p["roll_gain_yd_per_ms"] * h * np.cos(descent)
        roll = roll * p[f"skill.{skill}"] * p[f"roll.{lie}"]
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
