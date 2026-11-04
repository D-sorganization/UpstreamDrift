"""Impact conditions for optimiser loops (SHOW-C2, #14558).

A fast facade over the existing GCV-15/16 stack: a MuJoCo rollout goes through
:func:`clubhead_series_from_mujoco` and :func:`extract_impact_parameters`, and
the result is reduced to the five quantities an optimiser cost needs.  No new
kinematics lives here, so the sign conventions are exactly those of
``impact_parameters`` (face + = open, path + = in-to-out, AoA + = ascending).

Ball flight is deliberately not computed: the epic (#14556) uses it only as a
reporting layer through ``SwingBallFlightPipeline``, never inside the loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.contracts import ensure, require

from .adapters import NATIVE_CLUB_FACE, ClubFaceSpec, clubhead_series_from_mujoco
from .extract import ImpactParameters, extract_impact_parameters
from .target_frame import TargetFrame

_FIELDS = (
    "face_angle_deg",
    "club_path_deg",
    "attack_angle_deg",
    "clubhead_speed_mph",
    "face_to_path_deg",
    "dynamic_loft_deg",
    "toe_mm",
    "high_mm",
)


@dataclass(frozen=True)
class ImpactConditions:
    """Optimiser-facing impact conditions (degrees, mph, millimetres).

    Field names and units match :class:`ImpactParameters`.  ``None`` means
    unavailable and ``unavailable[name]`` carries the reason.
    """

    clubhead_speed_mph: float
    attack_angle_deg: float
    club_path_deg: float | None = None
    face_angle_deg: float | None = None
    face_to_path_deg: float | None = None
    dynamic_loft_deg: float | None = None
    toe_mm: float | None = None
    high_mm: float | None = None
    impact_time_s: float = 0.0
    unavailable: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_parameters(cls, params: ImpactParameters) -> ImpactConditions:
        """Reduce full :class:`ImpactParameters` to the optimiser quantities."""
        require(isinstance(params, ImpactParameters), "params must be ImpactParameters")
        values: dict[str, Any] = {name: getattr(params, name) for name in _FIELDS}
        return cls(
            impact_time_s=params.impact_time_s,
            unavailable={k: v for k, v in params.unavailable.items() if k in _FIELDS},
            **values,
        )

    def as_vector(self) -> tuple[float, float, float, float]:
        """``(face, path, aoa, speed_mph)`` for cost functions.

        Raises:
            ValueError: face or path is unavailable (never invented).
        """
        require(
            self.face_angle_deg is not None and self.club_path_deg is not None,
            "face and path must be available to build the cost vector",
        )
        vec = (
            float(self.face_angle_deg),  # type: ignore[arg-type]
            float(self.club_path_deg),  # type: ignore[arg-type]
            self.attack_angle_deg,
            self.clubhead_speed_mph,
        )
        ensure(all(math.isfinite(x) for x in vec), "conditions must be finite", vec)
        return vec


def evaluate_rollout(
    model: Any,
    times_s: object,
    qpos: object,
    qvel: object,
    body_name: str,
    frame: TargetFrame,
    *,
    spec: ClubFaceSpec = NATIVE_CLUB_FACE,
    impact_index: int | None = None,
    contact_index: int | None = None,
    ball_contact_m: object | None = None,
) -> ImpactConditions:
    """Impact conditions of one MuJoCo rollout, fast enough for optimiser loops.

    The impact state is the explicit ``impact_index``, the sample before
    ``contact_index``, or else the peak-clubhead-speed sample.  Tools D-plane
    estimates are skipped (``use_tools=False``): they are reporting extras.

    Raises:
        ValueError: invalid trajectory/indices or speed below 1 m/s.
    """
    require(isinstance(frame, TargetFrame), "frame must be a TargetFrame")
    series = clubhead_series_from_mujoco(model, times_s, qpos, qvel, body_name, spec)
    params = extract_impact_parameters(
        series,
        frame,
        impact_index=impact_index,
        contact_index=contact_index,
        use_tools=False,
        ball=_ball(ball_contact_m),
    )
    return ImpactConditions.from_parameters(params)


def _ball(contact_m: object | None):
    from .extract import BallObservation

    return None if contact_m is None else BallObservation(contact_m=contact_m)
