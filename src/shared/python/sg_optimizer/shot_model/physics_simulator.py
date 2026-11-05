"""Impact conditions -> landing offsets via the repo's swing/ball-flight pipeline.

Adapter for the dispersion tables (#14567): it maps one row of ``IMPACT_FIELDS``
(GCV-15 signs: +face open, +path in-to-out, +attack angle up; speed in mph,
angles in degrees, strike offsets in mm) to a ``SwingState`` and runs
``SwingBallFlightPipeline``. Frame: x downrange, y left, z up; results are
yards (``along`` = x, ``lateral`` = y, so an open face lands at negative y).

ASSUMPTION: club loft per skill is a typical value (``CLUB_LOFT_DEG``) and head
mass / MOI use the ``SwingState`` defaults; both are placeholders until a fit
(B7) or a club-spec source supplies them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.physics.ball_launch_conditions import EnvironmentalConditions
from src.shared.python.physics.swing_ball_flight_pipeline import (
    SwingBallFlightPipeline,
    SwingState,
)
from src.shared.python.sg_optimizer.shot_model.skill_family import (
    IMPACT_FIELDS,
    SKILL_NAMES,
)

MPH_TO_MS = 0.44704
M_TO_YD = 1.0 / 0.9144
CLUB_LOFT_DEG: dict[str, float] = {
    "driver": 10.5,
    "fairway_wood": 15.0,
    "long_iron": 21.0,
    "mid_iron": 28.0,
    "short_iron": 38.0,
    "wedge": 50.0,
    "pitch": 54.0,
    "chip": 56.0,
    "bunker": 56.0,
}


def swing_state_from_impact(impact: NDArray[np.float64], skill: str) -> SwingState:
    """One impact-condition row -> clubhead kinematics at impact."""
    require(skill in SKILL_NAMES, f"unknown skill {skill!r}")
    require(len(impact) == len(IMPACT_FIELDS), f"need {len(IMPACT_FIELDS)} values")
    speed, face, path, aoa, toe, high = (float(v) for v in impact)
    require(speed > 0, "clubhead speed must be > 0", speed)
    p, a, f = np.radians([path, aoa, face])
    direction = np.array(
        [np.cos(a) * np.cos(p), -np.cos(a) * np.sin(p), np.sin(a)], dtype=float
    )
    loft = np.radians(CLUB_LOFT_DEG[skill])
    normal = np.array(
        [np.cos(loft) * np.cos(f), -np.cos(loft) * np.sin(f), np.sin(loft)]
    )
    return SwingState(
        clubhead_velocity=direction * speed * MPH_TO_MS,
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=normal,
        clubhead_loft_deg=CLUB_LOFT_DEG[skill],
        impact_offset=np.array([toe, high]) / 1000.0,
        engine_name="sg_optimizer",
    )


class PipelineLandingSimulator:
    """``LandingSimulator`` over ``SwingBallFlightPipeline`` (one pipeline per wind)."""

    def __init__(self) -> None:
        self._pipelines: dict[tuple[float, float], SwingBallFlightPipeline] = {}

    def _pipeline(self, wind: tuple[float, float]) -> SwingBallFlightPipeline:
        if wind not in self._pipelines:
            head, cross = wind  # mph; headwind blows toward -x, crosswind toward +y
            env = EnvironmentalConditions(
                wind_velocity=np.array([-head, cross, 0.0]) * MPH_TO_MS
            )
            self._pipelines[wind] = SwingBallFlightPipeline(environment=env)
        return self._pipelines[wind]

    def __call__(
        self,
        impacts: NDArray[np.float64],
        skill: str,
        wind: tuple[float, float],
    ) -> NDArray[np.float64]:
        pipe = self._pipeline(wind)
        out = np.empty((len(impacts), 2))
        for i in range(len(impacts)):
            end = pipe.run(swing_state_from_impact(impacts[i], skill)).trajectory[-1]
            out[i] = (end.position[0] * M_TO_YD, end.position[1] * M_TO_YD)
        ensure(bool(np.isfinite(out).all()), "pipeline produced non-finite landing")
        return out
