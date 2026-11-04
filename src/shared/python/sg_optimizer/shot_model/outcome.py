"""``ShotOutcomeModel``: the pluggable seam between a player and where the ball lands.

Mirrors the ``putting_analytics`` ``PuttOutcomeModel`` protocol: the MDP and the
strategy layer depend on this protocol only, so the tour-baseline Gaussian
(always available, the quick fallback) and later physics-derived backends
(dispersion tables, #14567/#14568) are interchangeable by registry name.

Frames and units: yards, in the **aim frame** (``along`` is down the aim line,
``lateral`` is across it, as in ``TiltedBivariateGaussian``). Rotation into the
hole frame is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.shared.python.sg_optimizer.course.conditions import CourseConditions
    from src.shared.python.sg_optimizer.mdp.action import ShotAction
    from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile


@dataclass(frozen=True)
class LandingSamples:
    """``n`` sampled landing points of one shot, aim frame, yards."""

    along: NDArray[np.float64]
    lateral: NDArray[np.float64]

    def __post_init__(self) -> None:
        require(self.along.ndim == 1 and self.along.size > 0, "need a 1-D sample")
        require(self.along.shape == self.lateral.shape, "along/lateral shape mismatch")
        require(
            bool(np.all(np.isfinite(self.along)) and np.all(np.isfinite(self.lateral))),
            "landing samples must be finite",
        )

    def __len__(self) -> int:
        return int(self.along.size)


@runtime_checkable
class ShotOutcomeModel(Protocol):
    """Anything that can sample landing points for one shot of one player."""

    def sample(
        self,
        player: PlayerProfile,
        shot: ShotAction,
        lie: int,
        conditions: CourseConditions,
        rng: np.random.Generator,
        n: int,
    ) -> LandingSamples:
        """Return ``n`` landing points from starting ``lie`` (a ``LIE_CODES`` value).

        Randomness must come only from ``rng`` so equal seeds reproduce.
        """
        ...
