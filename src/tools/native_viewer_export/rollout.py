"""Roll and total distance after the carry of native flight exports (#15443).

The flight model ends at first ground contact (the carry). The roll comes from
the repository's existing runout model (``sg_optimizer.shot_model.
ground_runout``, id ``sg_runout_v2``), driven by the landing state and
landing backspin of the flight on a flat fairway: it is a labelled model assumption with documented
unmeasured coefficients, not a physical contact simulation. Total distance is
the carry plus the roll along the landing direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.core.physics_constants import YARD_TO_M
from src.shared.python.sg_optimizer.shot_model.ground_runout import (
    GROUND_MODEL_ID,
    GROUND_PARAMETERS,
    landing_backspin_rads,
    SlopeAwareGround,
)
from src.shared.python.sg_optimizer.shot_model.skill_fit import club_to_skill
from src.shared.python.sg_optimizer.shot_model.slope_wind_bins import SLOPE_BINS
from src.tools.native_viewer_export.display_units import DisplayUnits, distance_text

SETTINGS_PATH = Path(__file__).with_name("rollout_settings.json")
_LIES = tuple(k[5:] for k in GROUND_PARAMETERS if k.startswith("roll."))
_SKILLS = tuple(k[6:] for k in GROUND_PARAMETERS if k.startswith("skill."))


@dataclass(frozen=True)
class RolloutSettings:
    """Validated options of the roll (meanings: ``rollout_settings.json``)."""

    enabled: bool = True
    lie: str = "fairway"
    slope: str = "flat"
    default_skill: str = "driver"
    assumption: str = "rollout: flat-fairway model"
    hud_format: str = "Roll {roll}   Total {total}   ({assumption})"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be true or false")
        for name, allowed in (
            ("lie", _LIES),
            ("slope", tuple(SLOPE_BINS)),
            ("default_skill", _SKILLS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        try:
            self.hud_format.format(roll="1 yd", total="2 yd", assumption="x")
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            raise ValueError(f"hud_format is not usable: {exc!r}") from exc


def load_rollout_settings(path: Path | None = None) -> RolloutSettings:
    """Settings from ``path`` (default: the file beside this module).

    Raises ``ValueError`` for an unknown entry, lie, slope or skill.
    """
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    try:
        return RolloutSettings(**document)
    except TypeError as exc:
        if "argument" not in str(exc):
            raise
        raise ValueError(f"rollout settings: {exc}") from exc


@dataclass(frozen=True)
class Rollout:
    """Roll and total [m]; ``end_offset_m`` is the (down-range, left) roll vector."""

    roll_m: float
    total_m: float
    end_offset_m: tuple[float, float]
    skill: str
    lie: str
    slope: str
    assumption: str
    model_id: str = GROUND_MODEL_ID
    source: str = ""

    def hud_text(self, units: DisplayUnits, settings: RolloutSettings) -> str:
        """The HUD line: roll and total in ``units`` and the assumption label."""
        return settings.hud_format.format(
            roll=distance_text(self.roll_m, units),
            total=distance_text(self.total_m, units),
            assumption=self.assumption,
        )

    def record(self) -> dict[str, Any]:
        """Manifest entry: SI distances, the model, its inputs and its source."""
        return {
            "roll_m": self.roll_m,
            "total_m": self.total_m,
            "assumption": self.assumption,
            "model": self.model_id,
            "skill": self.skill,
            "lie": self.lie,
            "slope": self.slope,
            "source": self.source,
        }


def _source() -> str:
    """Provenance of the runout coefficients, from the model's own records."""
    sources = sorted({p.source for p in GROUND_PARAMETERS.values()})
    return f"{GROUND_MODEL_ID} coefficients: " + "; ".join(sources)


def _landing_state(flight: Any) -> np.ndarray:
    """``(1, 6)`` row (x, y yd; vx, vy, vz m/s; backspin rad/s) at landing.

    The velocity is the last finite difference of the path; the backspin is the
    launch spin decayed over the flight, times ``cos`` of the spin-axis tilt.
    """
    path = np.asarray(flight.trajectory_m, dtype=float)
    times = flight.times_s
    times = (
        np.linspace(0.0, flight.flight_time_s, len(path)) if times is None else times
    )
    step = float(times[-1] - times[-2])
    require(len(path) >= 2 and step > 0.0, "landing velocity needs two timed points")
    velocity = (path[-1] - path[-2]) / step
    xy = path[-1, :2] / float(YARD_TO_M)
    tilt = math.radians(flight.spin_axis_deg or 0.0)
    spin = landing_backspin_rads(flight.spin_rpm, flight.flight_time_s, math.cos(tilt))
    return np.array([[xy[0], xy[1], *velocity, spin]], dtype=float)


def compute_rollout(
    flight: Any, club: str | None, settings: RolloutSettings | None
) -> Rollout | None:
    """Roll and total of ``flight`` (a ``FlightSummary``) struck with ``club``.

    ``None`` when the roll is off, the flight is invalid or the club has no
    runout class. Postcondition: roll >= 0 and total = carry + roll.
    """
    if settings is None or not settings.enabled or not flight.valid:
        return None
    skill = settings.default_skill if club is None else club_to_skill(club)
    if skill not in _SKILLS:
        return None
    state = _landing_state(flight)
    rest = SlopeAwareGround()(state, settings.lie, settings.slope, skill)[0]
    offset = (rest - state[0, :2]) * float(YARD_TO_M)
    roll = math.hypot(float(offset[0]), float(offset[1]))
    ensure(math.isfinite(roll), "roll must be finite", roll)
    return Rollout(
        roll,
        float(flight.carry_m) + roll,
        (float(offset[0]), float(offset[1])),
        skill,
        settings.lie,
        settings.slope,
        settings.assumption,
        source=_source(),
    )
