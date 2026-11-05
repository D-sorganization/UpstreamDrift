"""Shot-shape ``ImpactTarget`` presets (SHOW-C6, #14562).

Right-handed convention: +face = open, +path = in-to-out.  The ball curves by
``face - path`` (negative = draw, positive = fade) and starts on the
D-plane blend ``W*face + (1-W)*path``.  A curving preset solves
``face - path = c`` and ``start = 0`` so the shot is aimed at the target:

    face = (1-W)*c,  path = -W*c      (c < 0 draw, c > 0 fade)

Everything here is an ASSUMPTION until sourced, and is listed on each preset:
the face weight ``W`` of the start line and the +-AoA of high/low.  The
speed target is left to the caller (C5 sets it).

Design by Contract:
    Preconditions: known preset name; non-negative finite curve magnitude.
    Postconditions: draw/fade start on the target line and mirror each other;
    high/low differ from straight only in angle of attack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.shared.python.contracts import require
from src.shared.python.impact_optimization.impact_target import ImpactTarget

#: ASSUMPTION: share of the ball start line set by the club face (D-plane
#: approximation for a driver); the rest comes from the path.
START_LINE_FACE_WEIGHT: float = 0.75
DEFAULT_CURVE_DEG: float = 4.0
#: ASSUMPTION: launch-bracketing AoA offsets (deg) for high / low presets.
HIGH_AOA_DEG: float = 3.0
LOW_AOA_DEG: float = -3.0

#: ASSUMPTION: |face-to-path| (deg) from which a shape is named "strong", and
#: below which it reads "straight".
STRONG_CURVE_DEG: float = 5.0
STRAIGHT_CURVE_DEG: float = 0.5

_NAMES = ("straight", "draw", "fade", "high", "low")
_W_ASSUME = "start line = 0.75*face + 0.25*path (D-plane approximation, unsourced)"


def shape_label(face_deg: float, path_deg: float) -> str:
    """Name a shot shape by face-to-path (D30), e.g. ``strong draw (... -6.0 deg)``.

    Face-to-path sets the curvature; a single angle never names the shape (the
    issue example face -2 / path +4 is face-to-path -6, a strong draw).
    """
    require(
        math.isfinite(face_deg) and math.isfinite(path_deg), "angles must be finite"
    )
    c = face_deg - path_deg
    if abs(c) < STRAIGHT_CURVE_DEG:
        word = "straight"
    else:
        word = ("strong " if abs(c) >= STRONG_CURVE_DEG else "") + (
            "draw" if c < 0 else "fade"
        )
    return f"{word} (face-to-path {c:+.1f} deg)"


@dataclass(frozen=True)
class ShotPreset:
    """A named target plus the assumptions it rests on."""

    name: str
    target: ImpactTarget
    assumptions: tuple[str, ...]

    @property
    def label(self) -> str:
        """Shape word and face-to-path value; high / low keep their attack angle.

        Straight / draw / fade defer to :func:`shape_label`, so a preset built
        with a non-default curve is named the same as its target would be.
        """
        t = self.target
        if self.name in ("high", "low"):
            return f"{self.name} (aoa {t.aoa_deg:+.1f} deg)"
        return shape_label(t.face_deg, t.path_deg)


def _curving(name: str, curve_deg: float) -> ShotPreset:
    w = START_LINE_FACE_WEIGHT
    c = curve_deg if name == "fade" else -curve_deg
    target = ImpactTarget(face_deg=(1.0 - w) * c, path_deg=-w * c)
    return ShotPreset(name, target, (_W_ASSUME,))


def preset(name: str, curve_deg: float = DEFAULT_CURVE_DEG) -> ShotPreset:
    """Build one preset; ``curve_deg`` is |face - path| for draw/fade."""
    require(name in _NAMES, f"unknown preset {name!r}; choose from {_NAMES}")
    require(
        math.isfinite(curve_deg) and curve_deg >= 0.0,
        "curve_deg must be finite and >= 0",
        curve_deg,
    )
    if name in ("draw", "fade"):
        return _curving(name, curve_deg)
    aoa = {"straight": 0.0, "high": HIGH_AOA_DEG, "low": LOW_AOA_DEG}[name]
    return ShotPreset(
        name,
        ImpactTarget(aoa_deg=aoa),
        (_W_ASSUME, "AoA offsets of +-3 deg are placeholders pending a source"),
    )


def all_presets(curve_deg: float = DEFAULT_CURVE_DEG) -> tuple[ShotPreset, ...]:
    return tuple(preset(n, curve_deg) for n in _NAMES)
