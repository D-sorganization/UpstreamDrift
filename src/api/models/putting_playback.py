"""Playback payloads for the putting-green routes (#14473).

The per-sample ball mode and the phase marks the playback jumps between,
and the three-dimensional playback response the R3F client draws.  The
models live here so ``src.api.routes.putting_green`` stays under the
monolith threshold; their OpenAPI names are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.shared.python.putting_dynamics.phases import phase_marks

if TYPE_CHECKING:
    from src.shared.python.putting_dynamics import TrajectorySample

BallModeName = Literal["airborne", "slide", "roll", "rest"]


class PhaseMarkResponse(BaseModel):
    """Where one run of a ball mode starts, for playback jumps (#14473)."""

    mode: BallModeName
    t_s: float
    index: int = Field(description="Index of the run's first sample")


def phase_marks_response(
    samples: Sequence[TrajectorySample],
) -> list[PhaseMarkResponse]:
    return [
        PhaseMarkResponse(mode=m.mode.value, t_s=m.t_s, index=m.index)
        for m in phase_marks(samples)
    ]


class Putt3DSampleResponse(BaseModel):
    """One frame in the three-dimensional playback trajectory."""

    t_s: float
    x_m: float
    y_m: float
    z_m: float
    speed_mps: float
    spin_rad_s: float
    mode: BallModeName


class Putt3DCollisionResponse(BaseModel):
    """Impact quantities displayed beside the slow-motion collision."""

    ball_speed_mps: float
    putter_speed_before_mps: float
    putter_speed_after_mps: float
    launch_angle_deg: float
    spin_rad_s: float
    impulse_n_s: float
    contact_time_proxy_s: float
    kinetic_energy_loss_j: float
    face_twist_rad_s: float
    twist_moment_n_m_s: float


class Putt3DSurfaceResponse(BaseModel):
    """Surface geometry metadata required by the R3F scene."""

    width_m: float
    height_m: float
    grade_percent: float
    downhill_aspect_deg: float
    hole_x_m: float
    hole_y_m: float


class Putt3DSimulationResponse(BaseModel):
    """Complete deterministic playback payload for the R3F client."""

    roll_model: str = Field(
        description="Roll model that produced this playback (ADR-0045 F1)"
    )
    samples: list[Putt3DSampleResponse]
    phases: list[PhaseMarkResponse] = Field(default_factory=list)
    collision: Putt3DCollisionResponse
    surface: Putt3DSurfaceResponse
    holed: bool
    total_distance_m: float
    duration_s: float
    skid_distance_m: float
