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
    path + in-to-out, AoA + ascending, toe + toward toe, high + toward crown).
    ``None`` means unavailable.
    """

    face_angle_deg: float | None
    club_path_deg: float | None
    attack_angle_deg: float
    clubhead_speed_mph: float
    toe_mm: float | None
    high_mm: float | None


#: Face is weighted this many times path by default (panel review, epic #14556).
FACE_TO_PATH_RATIO = 10.0


@dataclass(frozen=True)
class ImpactWeights:
    """Per-term weights; ``face / path`` defaults to 10:1 and is configurable."""

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
    toe_mm: float = 0.0
    high_mm: float = 0.0
    weights: ImpactWeights = field(default_factory=ImpactWeights)
    #: Strike error scale per unit weight (1 cm); angles and speed use 1 deg/mph.
    strike_scale_mm: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "face_deg",
            "path_deg",
            "aoa_deg",
            "toe_mm",
            "high_mm",
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
            math.isfinite(self.strike_scale_mm) and self.strike_scale_mm > 0.0,
            "strike_scale_mm must be positive and finite",
            self.strike_scale_mm,
        )

    def deviation(self, c: MeasuredImpact) -> float:
        """Weighted squared deviation of ``c`` from this target (0 on target).

        Preconditions: face and path available and all used values finite
        (never invented); an unavailable strike location skips the strike term.
        """
        require(
            c.face_angle_deg is not None and c.club_path_deg is not None,
            "face and path must be available to score a shot",
        )
        w = self.weights
        total = (
            w.face * (c.face_angle_deg - self.face_deg) ** 2
            + w.path * (c.club_path_deg - self.path_deg) ** 2
            + w.aoa * (c.attack_angle_deg - self.aoa_deg) ** 2
        )
        if c.toe_mm is not None and c.high_mm is not None:
            s = self.strike_scale_mm
            total += w.strike * (
                ((c.toe_mm - self.toe_mm) / s) ** 2
                + ((c.high_mm - self.high_mm) / s) ** 2
            )
        if self.speed_mph is not None:
            total += w.speed * (c.clubhead_speed_mph - self.speed_mph) ** 2
        require(math.isfinite(total), "measured impact must be finite", total)
        ensure(total >= 0.0, "deviation must be non-negative", total)
        return total
