"""Common random numbers for any ``ShotOutcomeModel`` (#14677).

Value iteration only converges when the Bellman operator is the same on every
sweep. A model that draws fresh samples per call makes the operator random, so
the solve wanders at the Monte Carlo noise floor. ``CommonRandomModel`` draws each
shot's landing samples once, from its own seeded generator, and replays them.
This is the model-side twin of the per-action offset cache in ``HoleMDP``
(``_offsets``, #14660).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.shot_model.outcome import (
    LandingSamples,
    ShotOutcomeModel,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.shared.python.sg_optimizer.course.conditions import CourseConditions
    from src.shared.python.sg_optimizer.mdp.action import ShotAction
    from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile


class CommonRandomModel:
    """Replay one fixed draw per (shot, starting lie, n) from the wrapped model.

    The ``rng`` passed to ``sample`` is ignored: randomness comes only from this
    wrapper's own ``seed``, so equal seeds reproduce. A wrapper serves one player
    and one set of conditions, because the cache key does not include them.
    """

    def __init__(self, model: ShotOutcomeModel, seed: int = 0) -> None:
        require(isinstance(model, ShotOutcomeModel), "model must be a ShotOutcomeModel")
        self._model = model
        self._rng = np.random.default_rng(seed)
        self._draws: dict[tuple[ShotAction, int, int], LandingSamples] = {}
        self._bound: tuple[PlayerProfile, CourseConditions] | None = None

    def sample(
        self,
        player: PlayerProfile,
        shot: ShotAction,
        lie: int,
        conditions: CourseConditions,
        rng: np.random.Generator,
        n: int,
    ) -> LandingSamples:
        """Return the cached draw for this shot, drawing it on first use."""
        if self._bound is None:
            self._bound = (player, conditions)
        require(
            self._bound[0] is player and self._bound[1] is conditions,
            "player and conditions are fixed for the life of a CommonRandomModel",
        )
        key = (shot, lie, n)
        if key not in self._draws:
            self._draws[key] = self._model.sample(
                player, shot, lie, conditions, self._rng, n
            )
        return self._draws[key]
