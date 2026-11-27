"""Tee under a lifted golf ball (#15150).

Strike placement (:mod:`model_appearance.ball_strike`) can leave the ball
standing 0.04 to 0.10 m above the ground. :func:`tee_for_ball` returns the peg
that holds it: a shaft from the ground and a wider cup under the ball. It is a
display object only: no collision, no mass, and it is placed once, so it stays
where it stands when the ball leaves. Every number comes from
``ball_tee_settings.json`` beside this module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SETTINGS_PATH = Path(__file__).with_name("ball_tee_settings.json")
Vector = tuple[float, float, float]
Rgba = tuple[float, float, float, float]


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


@dataclass(frozen=True)
class BallTeeSettings:
    """Validated ``ball_tee_settings.json`` (``ValueError`` when invalid)."""

    enabled: bool
    min_gap_m: float
    shaft_radius_m: float
    cup_radius_m: float
    cup_height_m: float
    rgba: Rgba

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false")
        _positive(self.min_gap_m, "min_gap_m")
        shaft = _positive(self.shaft_radius_m, "shaft_radius_m")
        cup = _positive(self.cup_radius_m, "cup_radius_m")
        _positive(self.cup_height_m, "cup_height_m")
        if cup < shaft:
            raise ValueError("cup_radius_m must be at least shaft_radius_m")
        colour = tuple(float(v) for v in self.rgba)
        if len(colour) != 4 or not all(0.0 <= v <= 1.0 for v in colour):
            raise ValueError("rgba must be four numbers from 0 to 1")
        object.__setattr__(self, "rgba", colour)


def load_ball_tee_settings(path: Path | None = None) -> BallTeeSettings:
    """Read the settings file; ``KeyError`` for a missing key, else as the class."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    keys = (
        "enabled",
        "min_gap_m",
        "shaft_radius_m",
        "cup_radius_m",
        "cup_height_m",
        "rgba",
    )
    return BallTeeSettings(*(raw[key] for key in keys))


@lru_cache(maxsize=1)
def default_ball_tee_settings() -> BallTeeSettings:
    """The shipped settings, read once (the renderer asks every frame)."""
    return load_ball_tee_settings()


@dataclass(frozen=True)
class TeePart:
    """Upright cylinder from ``base_z_m`` up ``length_m`` about the axis ``xy_m``."""

    xy_m: tuple[float, float]
    base_z_m: float
    length_m: float
    radius_m: float

    @property
    def top_z_m(self) -> float:
        return self.base_z_m + self.length_m

    @property
    def centre_m(self) -> Vector:
        return (self.xy_m[0], self.xy_m[1], self.base_z_m + 0.5 * self.length_m)


@dataclass(frozen=True)
class BallTee:
    """Shaft (``None`` when the cup reaches the ground), cup and colour.

    The tee has no collision and no mass: it describes render geometry only.
    """

    shaft: TeePart | None
    cup: TeePart
    rgba: Rgba

    @property
    def parts(self) -> tuple[TeePart, ...]:
        return (self.cup,) if self.shaft is None else (self.shaft, self.cup)


def tee_for_ball(
    centre_m: Sequence[float],
    radius_m: float,
    settings: BallTeeSettings,
    ground_height_m: float = 0.0,
) -> BallTee | None:
    """The tee under a ball, or ``None`` when none is wanted.

    ``None`` when the tee is disabled or the underside of the ball is not more
    than ``min_gap_m`` above the ground (a ball on the turf).

    Preconditions (``ValueError``): ``centre_m`` a finite 3-vector, positive
    finite ``radius_m``, finite ground, ``cup_radius_m`` below ``radius_m``.
    Postconditions: the shaft base is on the ground; the cup top is where the
    cup rim meets the ball, at most one sagitta (under 4 mm for the default
    cup) above the lowest point of the ball; the cup is cut to the tee height.
    """
    x, y, z = (float(v) for v in centre_m)
    radius = _positive(radius_m, "radius_m")
    if not all(map(math.isfinite, (x, y, z, ground_height_m))):
        raise ValueError("centre_m and ground_height_m must be finite")
    if settings.cup_radius_m >= radius:
        raise ValueError("cup_radius_m must be smaller than the ball radius")
    if not settings.enabled or z - radius - ground_height_m <= settings.min_gap_m:
        return None
    top = z - math.sqrt(radius**2 - settings.cup_radius_m**2)
    height = min(settings.cup_height_m, top - ground_height_m)
    cup = TeePart((x, y), top - height, height, settings.cup_radius_m)
    shaft_length = cup.base_z_m - ground_height_m
    shaft = (
        TeePart((x, y), ground_height_m, shaft_length, settings.shaft_radius_m)
        if shaft_length > 1e-9
        else None
    )
    return BallTee(shaft, cup, settings.rgba)
