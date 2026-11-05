"""``ImpactTarget``: desired impact conditions with face-critical weighting.

The default target is the straight shot the owner asked for: face 0, path 0,
level angle of attack, with the face weighted 10:1 over the path (a square
face matters far more than a square path for where the ball starts and how it
curves).  The deviation is a weighted sum of squares of scaled errors, so it is
zero at the target, symmetric and monotone in each error.

Design by Contract:
    Preconditions: finite target values, positive weights/scales, speed (if
    given) positive.
    Postconditions: ``deviation`` is finite and non-negative; zero only when
    every weighted term is on target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Protocol

from src.shared.python.contracts import ensure, require


class MeasuredImpact(Protocol):
    """Structural view of ``impact_parameters.conditions.ImpactConditions`` (C2).

    Matching by attribute name keeps C1 free of an import dependency on the C2
    evaluator while sharing its single type (sign conventions: face + open,
    path + in-to-out, AoA + ascending, x + toward toe, y + toward crown).
    ``None`` means unavailable.
    """

    face_deg: float | None
    path_deg: float | None
    aoa_deg: float
    speed_mph: float
    strike_offset_x_m: float | None
    strike_offset_y_m: float | None


#: Face is weighted this many times path by default (panel review, epic #14556).
FACE_TO_PATH_RATIO = 10.0


@dataclass(frozen=True)
class ImpactWeights:
    """Per-term weights; ``face / path`` defaults to 10:1 and is configurable.

    The cost is quadratic, so 10:1 is a *weight* ratio: at iso-cost a 1 deg
    face error equals ``sqrt(10)`` ~ 3.16 deg of path error.  Supervisor ruling
    D10 keeps this reading as the default.  The other reading, 10:1 in degrees
    (1 deg face == 10 deg path), is a weight ratio of 100
    (``ImpactWeights(face=100.0)``) and is deliberately NOT the default: ball
    curvature follows face - path, so at 100:1 the optimiser could trade a
    near-zero face error for a large path error and bend a "straight" target
    into a draw or fade.  10:1 still keeps the face far dominant.  Report the
    face-to-path gap alongside the cost so the trade-off stays visible.
    """

    face: float = FACE_TO_PATH_RATIO
    path: float = 1.0
    aoa: float = 1.0
    speed: float = 1.0
    strike: float = 1.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            require(
                math.isfinite(value) and value > 0.0,
                f"weight {f.name} must be positive and finite",
                value,
            )


@dataclass(frozen=True)
class ImpactTarget:
    """Target impact conditions.  ``speed_mph=None`` leaves speed unconstrained."""

    face_deg: float = 0.0
    path_deg: float = 0.0
    aoa_deg: float = 0.0
    speed_mph: float | None = None
    strike_offset_x_m: float = 0.0
    strike_offset_y_m: float = 0.0
    weights: ImpactWeights = field(default_factory=ImpactWeights)
    #: Strike error scale per unit weight (1 cm); angles and speed use 1 deg/mph.
    strike_scale_m: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "face_deg",
            "path_deg",
            "aoa_deg",
            "strike_offset_x_m",
            "strike_offset_y_m",
        ):
            value = getattr(self, name)
            require(math.isfinite(value), f"{name} must be finite", value)
        if self.speed_mph is not None:
            require(
                math.isfinite(self.speed_mph) and self.speed_mph > 0.0,
                "speed_mph must be positive and finite",
                self.speed_mph,
            )
        require(
            math.isfinite(self.strike_scale_m) and self.strike_scale_m > 0.0,
            "strike_scale_m must be positive and finite",
            self.strike_scale_m,
        )

    def deviation(self, c: MeasuredImpact) -> float:
        """Weighted squared deviation of ``c`` from this target (0 on target).

        Preconditions: face and path available and all used values finite
        (never invented); an unavailable strike location skips the strike term.
        """
        face, path = c.face_deg, c.path_deg
        require(
            face is not None and path is not None,
            "face and path must be available to score a shot",
        )
        assert face is not None and path is not None  # narrowing for mypy
        w = self.weights
        total = (
            w.face * (face - self.face_deg) ** 2
            + w.path * (path - self.path_deg) ** 2
            + w.aoa * (c.aoa_deg - self.aoa_deg) ** 2
        )
        if c.strike_offset_x_m is not None and c.strike_offset_y_m is not None:
            s = self.strike_scale_m
            total += w.strike * (
                ((c.strike_offset_x_m - self.strike_offset_x_m) / s) ** 2
                + ((c.strike_offset_y_m - self.strike_offset_y_m) / s) ** 2
            )
        if self.speed_mph is not None:
            total += w.speed * (c.speed_mph - self.speed_mph) ** 2
        require(math.isfinite(total), "measured impact must be finite", total)
        ensure(total >= 0.0, "deviation must be non-negative", total)
        return total
