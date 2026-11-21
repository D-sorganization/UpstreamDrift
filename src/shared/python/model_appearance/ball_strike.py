"""Ball placed where the club face strikes it (#15150).

A replayed swing does not bring the face centre back to the address ball
(:mod:`model_appearance.ball`): a driver ball is teed higher, and a fit has its
own error. :func:`strike_ball_centre` moves the ball to where the face centre
passes nearest it. This is a display placement only: no joint angle or club
dimension changes and the face is not squared. Every number comes from
``ball_strike_settings.json`` beside this module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.shared.python.model_appearance.club_face import nearest_on_path

SETTINGS_PATH = Path(__file__).with_name("ball_strike_settings.json")
PLACEMENTS = ("address", "strike")
DEFAULT_CLUB = "default"
Vector = tuple[float, float, float]


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


@dataclass(frozen=True)
class BallStrikeSettings:
    """Validated ``ball_strike_settings.json`` (``ValueError`` when invalid)."""

    placement: str
    search_window_s: float
    max_shift_m: float
    max_tee_height_m: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}")
        _positive(self.search_window_s, "search_window_s")
        _positive(self.max_shift_m, "max_shift_m")
        limits = {str(k): float(v) for k, v in dict(self.max_tee_height_m).items()}
        if DEFAULT_CLUB not in limits:
            raise ValueError(f"max_tee_height_m needs a {DEFAULT_CLUB!r} entry")
        if not all(math.isfinite(v) and v >= 0.0 for v in limits.values()):
            raise ValueError("max_tee_height_m values must be finite and not negative")
        object.__setattr__(self, "max_tee_height_m", limits)

    def tee_limit_m(self, club: str) -> float:
        """Greatest tee height of ``club`` (its head name), else the default."""
        return self.max_tee_height_m.get(club, self.max_tee_height_m[DEFAULT_CLUB])


def load_ball_strike_settings(path: Path | None = None) -> BallStrikeSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    keys = ("placement", "search_window_s", "max_shift_m", "max_tee_height_m")
    return BallStrikeSettings(*(raw[key] for key in keys))


@dataclass(frozen=True)
class StrikePlacement:
    """Ball centre at the strike, its move from the address ball (``shift_m``)
    and the vertical move the ground or tee limit forced (``height_limited_m``:
    0 for a strike at the face centre, negative when the ball was held below)."""

    centre_m: Vector
    contact_time_s: float
    shift_m: Vector
    tee_height_m: float
    height_limited_m: float

    def to_dict(self) -> dict[str, Any]:
        """The placement as a JSON record (lists, not tuples)."""
        record = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in record.items()}


def _path(times_s: object, centre_m: object, normal: object) -> tuple[Any, Any, Any]:
    t = np.asarray(times_s, dtype=float)
    centre = np.asarray(centre_m, dtype=float)
    unit = np.asarray(normal, dtype=float)
    if t.ndim != 1 or t.shape[0] < 2:
        raise ValueError("the face path needs at least 2 samples")
    if centre.shape != (t.shape[0], 3) or unit.shape != centre.shape:
        raise ValueError(f"face centres and normals must have shape ({t.shape[0]}, 3)")
    if not (np.isfinite(t).all() and np.isfinite(centre).all()):
        raise ValueError("face path times and centres must be finite")
    if not np.isfinite(unit).all() or np.any(np.linalg.norm(unit, axis=1) < 1e-9):
        raise ValueError("face normals must be finite and nonzero")
    return t, centre, unit


def _nearest_pass(t: Any, centre: Any, normal: Any, point: Any) -> tuple[Any, ...]:
    """Time, face centre and unit normal where the sampled path passes ``point``."""
    along, foot = nearest_on_path(centre, point)
    i = int(np.argmin(np.linalg.norm(foot - point, axis=1)))
    s = float(along[i])
    blend = (1.0 - s) * normal[i] + s * normal[i + 1]
    return float(t[i] + s * (t[i + 1] - t[i])), foot[i], blend / np.linalg.norm(blend)


def strike_ball_centre(
    times_s: object,
    face_centre_m: object,
    face_normal: object,
    nominal_m: object,
    *,
    max_tee_height_m: float,
    max_shift_m: float,
    ground_height_m: float = 0.0,
    radius_m: float = BALL_RADIUS_M,
) -> StrikePlacement:
    """Ball centre that the sampled face path strikes at the face centre.

    The contact point is the point of the path (world, Z up, straight lines
    between samples) nearest ``nominal_m``, the address ball centre; the ball
    centre lies one radius ahead of it along the face normal, then is held
    between the ground and the tee limit.

    Preconditions (``ValueError``): at least 2 samples, finite ``(N, 3)``
    arrays, nonzero normals, a finite 3-vector ``nominal_m``, positive
    ``radius_m`` and ``max_shift_m``, ``max_tee_height_m`` not negative, and a
    move of at most ``max_shift_m``. Postcondition: with ``height_limited_m ==
    0`` the ball touches the face at its centre at ``contact_time_s``.
    """
    t, centre, normal = _path(times_s, face_centre_m, face_normal)
    nominal = np.asarray(nominal_m, dtype=float)
    if nominal.shape != (3,) or not np.isfinite(nominal).all():
        raise ValueError("nominal_m must be a finite 3-vector")
    radius = _positive(radius_m, "radius_m")
    limit = _positive(max_shift_m, "max_shift_m")
    tee = float(max_tee_height_m)
    if not math.isfinite(tee) or tee < 0.0 or not math.isfinite(ground_height_m):
        raise ValueError("max_tee_height_m must be >= 0 and the ground finite")
    when, contact, unit = _nearest_pass(t, centre, normal, nominal)
    ball = contact + radius * unit
    floor = ground_height_m + radius
    free_height = float(ball[2])
    ball[2] = min(max(free_height, floor), floor + tee)
    shift = ball - nominal
    moved = float(np.linalg.norm(shift))
    if moved > limit:
        raise ValueError(
            f"the face centre passes {moved:.3f} m from the address ball, more than "
            f"the {limit:.3f} m limit: the swing does not come back to the ball"
        )
    x, y, z = map(float, ball)
    dx, dy, dz = map(float, shift)
    return StrikePlacement((x, y, z), when, (dx, dy, dz), z - floor, z - free_height)
