"""Bounce and roll after carry, for the SHOW-B3b dispersion tables (#14610).

The flight pipeline ends at the first ground contact (carry).  This adds the
rest position: a landing state ``(x, y, vx, vy, vz, spin)`` (yards, then m/s in
the aim frame: x downrange, y left, z up; ``vz < 0`` descending; ``spin`` the
backspin at landing in rad/s) becomes resting offsets.  Ground model
``sg_runout_v2``, per shot:

* **bounce** = the first hop of Penner's spin-aware turf bounce (Can. J. Phys. 80,
  931-940, 2002, eqs. 2-8; coefficients cited in ``src/config/runout_bounce.json``):
  the turf is a rigid surface tilted by ``theta_c`` (eq. 8), the ball rolls out
  (eq. 3, ``vx' = 5/7 vx' - 2/7 r w``) or slides (eq. 2) by the critical friction of
  eq. 4, and the hop is the drag-free range ``2 vx vz / g`` of the rebound, along the
  horizontal landing direction; more backspin hops shorter (or backwards);
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
rough in these tables); ball-above/below-feet bins have no runout effect.  The
``bounce.*`` coefficients are cited; every other coefficient is a labelled
assumption (``GROUND_PARAMETERS``).  Penner fitted ``theta_c`` to one green impact
and ``e`` on soft turf, and the bounce is applied to every lie unchanged.  Green-side skills (pitch, chip, bunker) keep carry: they land outside this fairway-turf
domain (``CARRY_ONLY_SKILLS``, hashed).  Credit: first drafted in Cursor PR #14642.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.core.physics_constants import (
    GRAVITY_M_S2,
    SPIN_DECAY_RATE_S,
    YARD_TO_M,
)
from src.shared.python.physics.lie_interaction import ParameterRecord
from src.shared.python.sg_optimizer.shot_model.slope_wind_bins import SLOPE_BINS

GROUND_MODEL_ID = "sg_runout_v2"
LANDING_COLUMNS = ("x_yd", "y_yd", "vx_ms", "vy_ms", "vz_ms", "spin_rads")
BOUNCE_SETTINGS_PATH = Path(__file__).resolve().parents[4] / "config/runout_bounce.json"
_ASSUMED = "Assumed prior; unmeasured turf, no primary citation"


def _p(value: float, why: str) -> ParameterRecord:
    return ParameterRecord(value, "assumption", _ASSUMED, why)


def load_bounce_settings() -> dict[str, Any]:
    """The cited first-bounce coefficients (``runout_bounce.json``)."""
    raw: dict[str, Any] = json.loads(BOUNCE_SETTINGS_PATH.read_text(encoding="utf-8"))
    require(bool(raw.get("parameters")), "bounce settings need parameters")
    return raw


def _cited() -> dict[str, ParameterRecord]:
    raw = load_bounce_settings()
    src = raw["source"]
    cite = (
        f"{src['authors']} ({src['year']}) {src['title']}, {src['venue']} "
        f"{src['volume']}, {src['pages']}, doi:{src['doi']}"
    )
    return {
        f"bounce.{k}": ParameterRecord(float(r["value"]), "cited", cite, r["where"])
        for k, r in raw["parameters"].items()
    }


GROUND_PARAMETERS: dict[str, ParameterRecord] = {
    **_cited(),
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


@dataclass(frozen=True)
class Rebound:
    """Penner first-bounce rebound: velocity (m/s, x along the landing, z up)."""

    vx: NDArray[np.float64]
    vz: NDArray[np.float64]
    mu_c: NDArray[np.float64]  # eq. (4)
    e: NDArray[np.float64]  # eq. (5)
    rolling: NDArray[np.bool_]  # mu >= |mu_c|: eq. (3), else eq. (2)


def penner_rebound(horizontal: Any, down: Any, spin: Any) -> Rebound:
    """Rebound of a ball landing at ``horizontal`` and ``down`` (> 0) m/s with
    ``spin`` rad/s of backspin, by Penner (2002) eqs. (2)-(8)."""
    h, d, w = np.broadcast_arrays(
        *(np.asarray(a, float) for a in (horizontal, down, spin))
    )
    require(bool(np.all(d > 0.0) and np.all(h >= 0.0)), "landing must descend")
    require(bool(np.isfinite(w).all()), "landing spin must be finite")
    p = {k[7:]: v.value for k, v in GROUND_PARAMETERS.items() if k[:7] == "bounce."}
    phi = np.degrees(np.arctan2(h, d))  # impact angle from vertical
    tc = np.radians(
        p["theta_c_ref_deg"]
        * (np.hypot(h, d) / p["theta_c_ref_speed_mps"])
        * (phi / p["theta_c_ref_angle_deg"])
    )  # eq. (8)
    tc = np.minimum(tc, np.pi / 4)  # calibrated at 15.4 deg; extrapolation past 45 deg is unphysical
    c, s = np.cos(tc), np.sin(tc)
    vix, viy = h * c - d * s, h * s + d * c  # eq. (6)
    e = np.where(
        viy <= p["restitution_speed_limit_mps"],
        p["restitution_c0"]
        - p["restitution_c1_s_per_m"] * viy
        + p["restitution_c2_s2_per_m2"] * viy**2,
        p["restitution_high_speed"],
    )  # eq. (5)
    rw = p["ball_radius_m"] * w
    mu_c = 2.0 * (vix + rw) / (7.0 * (1.0 + e) * viy)  # eq. (4)
    rolling = p["friction_mu"] >= np.abs(mu_c)
    vrx = np.where(
        rolling,
        5.0 / 7.0 * vix - 2.0 / 7.0 * rw,
        vix - np.sign(mu_c) * p["friction_mu"] * viy * (1 + e),
    )  # eqs. (3a), (2a)
    vry = e * viy  # eqs. (3b), (2b)
    return Rebound(vrx * c - vry * s, vrx * s + vry * c, mu_c, e, rolling)  # eq. (7)


def first_hop_m(horizontal: Any, down: Any, spin: Any) -> NDArray[np.float64]:
    """Signed drag-free range (m) of the first Penner hop on flat ground."""
    out = penner_rebound(horizontal, down, spin)
    return 2.0 * out.vx * np.maximum(out.vz, 0.0) / float(GRAVITY_M_S2)


def landing_backspin_rads(
    spin_rpm: float, flight_time_s: float, backspin_fraction: float = 1.0
) -> float:
    """Backspin (rad/s) at landing: the launch spin decayed at the kernel's rate
    (``SPIN_DECAY_RATE_S``, ``exp(-k t)``), times the share about the backspin axis."""
    require(flight_time_s >= 0.0 and np.isfinite(spin_rpm), "bad spin or time")
    decay = np.exp(-float(SPIN_DECAY_RATE_S) * flight_time_s)
    return float(spin_rpm * 2.0 * np.pi / 60.0 * decay * backspin_fraction)


@dataclass(frozen=True)
class SlopeAwareGround:
    """``GroundModel``: landing states ``(n, 6)`` -> resting offsets ``(n, 2)`` yards."""

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
        vx, vy, vz, spin = landing[:, 2], landing[:, 3], landing[:, 4], landing[:, 5]
        h = np.hypot(vx, vy)
        descent = np.arctan2(-vz, h).clip(np.radians(5.0), np.radians(85.0))
        grade = np.tan(np.radians(SLOPE_BINS[slope].stance_deg))
        slope_mult = max(_MIN_SLOPE_MULT, 1.0 - p["slope_roll_per_tan"] * grade)
        speed = np.hypot(h, vz)
        hop = first_hop_m(speed * np.cos(descent), speed * np.sin(descent), spin)
        bounce = hop / float(YARD_TO_M)
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
