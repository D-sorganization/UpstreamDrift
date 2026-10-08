"""Club-ball collision impulse, shared by every engine (GCV-20 #11767, GCV-13).

One definition of what the ball does to the club. A ball at rest is struck
by a clubhead whose face centre approaches along the unit face normal ``n``
at ``v_n = v . n > 0``. The clubhead is represented by its *effective mass*
along the normal at the contact point, ``m = 1 / (n^T J M^-1 J^T n)``, which
the caller derives from the articulated plant (the club is not free; the
shaft and arms add inertia). Frictionless contact, normal restitution ``e``:

    J = (1 + e) m m_b v_n / (m + m_b)          (impulse magnitude)
    impulse on ball = J n,   impulse on club = -J n
    club normal speed change = J / m,  ball launch speed = J / m_b

The impulse is delivered as a constant force over the contact duration.

Constants and sources
---------------------
* ``BALL_MASS_KG`` 0.04593 kg: Rules of Golf Equipment Standards (R&A/USGA),
  ball mass not greater than 45.93 g.
* ``COR_LIMIT`` 0.830: R&A/USGA clubhead coefficient-of-restitution limit
  for drivers. Used as a constant (no confirmed speed-dependent source is
  in the repository). It is a ceiling: irons are lower, so the iron rebound
  is an upper bound.
* ``CONTACT_DURATION_S`` 0.45 ms: Cochran & Stobbs, *The Search for the
  Perfect Swing* (1968), about half a millisecond of contact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.ball import BALL_RADIUS_M

BALL_MASS_KG: float = 0.04593
COR_LIMIT: float = 0.830
CONTACT_DURATION_S: float = 4.5e-4

Vector = Sequence[float] | NDArray[np.float64]


@dataclass(frozen=True)
class BallCollision:
    """Result of one club-ball collision (all vectors world frame, SI)."""

    face_normal: NDArray[np.float64]
    application_point_m: NDArray[np.float64]
    effective_mass_kg: float
    ball_mass_kg: float
    cor: float
    duration_s: float
    time_s: float
    approach_speed_mps: float
    impulse_on_ball_n_s: NDArray[np.float64]
    contact_point_m: NDArray[np.float64] | None

    @property
    def impulse_on_club_n_s(self) -> NDArray[np.float64]:
        """Equal and opposite to the impulse on the ball."""
        return -self.impulse_on_ball_n_s

    @property
    def impulse_magnitude_n_s(self) -> float:
        return float(np.linalg.norm(self.impulse_on_ball_n_s))

    @property
    def club_normal_speed_change_mps(self) -> float:
        return self.impulse_magnitude_n_s / self.effective_mass_kg

    @property
    def ball_speed_mps(self) -> float:
        return self.impulse_magnitude_n_s / self.ball_mass_kg

    @property
    def mean_force_on_club_n(self) -> NDArray[np.float64]:
        return self.impulse_on_club_n_s / self.duration_s

    @property
    def face_to_ball_gap_m(self) -> float | None:
        """Distance from the force application point to the ball surface."""
        if self.contact_point_m is None:
            return None
        gap = self.application_point_m - self.contact_point_m
        return float(np.linalg.norm(gap))

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable record for receipts and bundle provenance."""

        def vec(x: NDArray[np.float64] | None) -> list[float] | None:
            return None if x is None else [float(c) for c in x]

        return {
            "model": "frictionless normal impulse, constant COR",
            "time_s": self.time_s,
            "duration_s": self.duration_s,
            "ball_mass_kg": self.ball_mass_kg,
            "cor": self.cor,
            "effective_mass_kg": self.effective_mass_kg,
            "approach_speed_mps": self.approach_speed_mps,
            "face_normal": vec(self.face_normal),
            "application_point_m": vec(self.application_point_m),
            "impulse_on_club_n_s": vec(self.impulse_on_club_n_s),
            "club_normal_speed_change_mps": self.club_normal_speed_change_mps,
            "ball_speed_mps": self.ball_speed_mps,
            "contact_point_m": vec(self.contact_point_m),
            "face_to_ball_gap_m": self.face_to_ball_gap_m,
        }


def _vector3(value: Vector, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return arr


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return number


def validate_window(
    time_s: float, duration_s: float, swing_span_s: tuple[float, float]
) -> None:
    """Require ``[time_s, time_s + duration_s]`` inside the swing span."""
    start, end = (float(swing_span_s[0]), float(swing_span_s[1]))
    if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
        raise ValueError(f"swing_span_s must increase, got {swing_span_s!r}")
    if not np.isfinite(time_s) or time_s < start or time_s + duration_s > end:
        raise ValueError(
            f"impact window [{time_s}, {time_s + duration_s}] s must lie "
            f"inside the swing [{start}, {end}] s"
        )


def collision_impulse(
    *,
    face_normal: Vector,
    face_velocity_mps: Vector,
    application_point_m: Vector,
    effective_mass_kg: float,
    time_s: float,
    swing_span_s: tuple[float, float],
    ball_mass_kg: float = BALL_MASS_KG,
    cor: float = COR_LIMIT,
    duration_s: float = CONTACT_DURATION_S,
    ball_centre_m: Vector | None = None,
) -> BallCollision:
    """Impulse exchanged between a ball at rest and the approaching face.

    Preconditions (``ValueError``): positive finite masses and duration;
    ``0 < cor <= 1``; nonzero face normal; the face moving into the ball
    (``v . n > 0``); finite vectors; the contact window inside the swing.
    Postconditions: the club impulse is ``-J n``; normal momentum is
    conserved and the post-collision separation speed is ``cor * v_n``.
    """
    m_eff = _positive(effective_mass_kg, "effective_mass_kg")
    m_ball = _positive(ball_mass_kg, "ball_mass_kg")
    restitution = float(cor)
    if not np.isfinite(restitution) or not 0.0 < restitution <= 1.0:
        raise ValueError(f"cor must lie in (0, 1], got {cor!r}")
    duration = _positive(duration_s, "duration_s")
    validate_window(float(time_s), duration, swing_span_s)
    normal = _vector3(face_normal, "face_normal")
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        raise ValueError("face_normal must be nonzero")
    normal = normal / length
    velocity = _vector3(face_velocity_mps, "face_velocity_mps")
    point = _vector3(application_point_m, "application_point_m")
    v_n = float(velocity @ normal)
    if v_n <= 0.0:
        raise ValueError(
            f"the face must be moving into the ball (v . n = {v_n:.3f} m/s)"
        )
    magnitude = (1.0 + restitution) * m_eff * m_ball * v_n / (m_eff + m_ball)
    contact = None
    if ball_centre_m is not None:
        contact = _vector3(ball_centre_m, "ball_centre_m") - BALL_RADIUS_M * normal
    return BallCollision(
        face_normal=normal,
        application_point_m=point,
        effective_mass_kg=m_eff,
        ball_mass_kg=m_ball,
        cor=restitution,
        duration_s=duration,
        time_s=float(time_s),
        approach_speed_mps=v_n,
        impulse_on_ball_n_s=magnitude * normal,
        contact_point_m=contact,
    )
