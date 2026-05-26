"""MDP action primitives.

A Phase-1 action is (club, aim_angle_radians). Shot types (knockdown / draw /
fade) are deferred to Phase 2+ per spec §1.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require


@dataclass(frozen=True)
class ShotAction:
    """A single (club, aim) decision."""

    club: str
    aim_angle_rad: float


@dataclass(frozen=True)
class ActionSet:
    """Discretized action set: clubs × aim grid."""

    clubs: tuple[str, ...]
    aim_grid_deg: NDArray[np.float64] = field(
        default_factory=lambda: np.linspace(-45.0, 45.0, 31)
    )

    def __post_init__(self) -> None:
        require(len(self.clubs) > 0, "at least one club required")
        require(len(self.aim_grid_deg) > 0, "non-empty aim grid required")

    @property
    def aim_grid_rad(self) -> NDArray[np.float64]:
        return np.deg2rad(self.aim_grid_deg)

    def iter_actions(self):
        for c in self.clubs:
            for a in self.aim_grid_rad:
                yield ShotAction(club=c, aim_angle_rad=float(a))


def default_action_set(include_putter: bool = False) -> ActionSet:
    # The through-the-bag MDP treats putts via the putting model (spec §1.3);
    # the putter is intentionally not part of the swing-action set.
    clubs: tuple[str, ...] = (
        "driver",
        "3_wood",
        "5_iron",
        "7_iron",
        "9_iron",
        "pw",
        "sw",
        "lw",
    )
    if include_putter:
        clubs = clubs + ("putter",)
    # 5° resolution is the spec floor (§6 pitfall #4 calls 5° marginal); we use
    # 3° at the centre and coarser at the wings.
    deg = np.concatenate(
        [
            np.linspace(-30.0, -12.0, 7),
            np.linspace(-9.0, 9.0, 13),
            np.linspace(12.0, 30.0, 7),
        ]
    )
    return ActionSet(clubs=clubs, aim_grid_deg=deg)


# Used by the solver to translate angles consistently with the shot model.
def rotate(dx: float, dy: float, angle_rad: float) -> tuple[float, float]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return c * dx - s * dy, s * dx + c * dy
