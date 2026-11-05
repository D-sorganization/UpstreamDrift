"""Tour-baseline ``ShotOutcomeModel`` backend (the always-available fallback).

Behaviour-preserving adaptation of the Gaussian + ``BaselineBag`` path that
``mdp.transition`` used inline before #14565.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES
from src.shared.python.sg_optimizer.shot_model.outcome import LandingSamples

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.shared.python.sg_optimizer.course.conditions import CourseConditions
    from src.shared.python.sg_optimizer.mdp.action import ShotAction
    from src.shared.python.sg_optimizer.shot_model.baseline import BaselineBag
    from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile

_TREES = LIE_CODES["trees"]
_ROUGH = LIE_CODES["rough"]
_SAND = LIE_CODES["sand"]


def lie_modifiers(
    start_lie: int, conditions: CourseConditions
) -> tuple[float, float, float]:
    """Return (distance_mult, sigma_long_mult, sigma_lat_mult) from starting lie."""
    if start_lie == _ROUGH:
        return (
            conditions.rough.distance_multiplier(),
            conditions.rough.dispersion_multiplier(),
            conditions.rough.dispersion_multiplier(),
        )
    if start_lie == _TREES:
        return (
            conditions.trees.distance_multiplier(),
            conditions.trees.dispersion_multiplier(),
            conditions.trees.dispersion_multiplier(),
        )
    if start_lie == _SAND:
        return (0.85, 1.30, 1.30)  # fixed sand penalty; tuneable later
    return (1.0, 1.0, 1.0)


class BaselineShotModel:
    """Tilted-Gaussian dispersion around the player's effective carry."""

    def __init__(self, baseline: BaselineBag) -> None:
        self._baseline = baseline

    def sample(
        self,
        player: PlayerProfile,
        shot: ShotAction,
        lie: int,
        conditions: CourseConditions,
        rng: np.random.Generator,
        n: int,
    ) -> LandingSamples:
        require(n > 0, "n must be > 0", n)
        dist_mult, sl_mult, slat_mult = lie_modifiers(lie, conditions)
        dist = player.effective_distance(shot.club, self._baseline) * dist_mult
        distr = player.effective_distribution(shot.club, self._baseline).scaled(
            sl_mult, slat_mult
        )
        offsets = distr.sample(n, rng)  # (n, 2): along, lateral
        out = LandingSamples(along=dist + offsets[:, 0], lateral=offsets[:, 1])
        ensure(len(out) == n, "backend must return exactly n samples")
        return out
