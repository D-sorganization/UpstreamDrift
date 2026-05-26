"""Stochastic transitions — sample shot outcomes under player + conditions.

For a (state, action) the transition is:
  1. Look up the player's effective dispersion for the chosen club, modulated
     by the **starting** lie via ``CourseConditions`` (pitfall #12).
  2. Sample carry/lateral offsets in the aim frame.
  3. Rotate into the hole frame; add to the current ball position.
  4. Look up the landing lie code from the raster.
  5. Apply hazard rules: water/OB → drop with stroke penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LieRaster
from src.shared.python.sg_optimizer.mdp.action import ShotAction
from src.shared.python.sg_optimizer.mdp.state import State

if TYPE_CHECKING:  # pragma: no cover
    from src.shared.python.sg_optimizer.course.conditions import CourseConditions
    from src.shared.python.sg_optimizer.shot_model.baseline import BaselineBag
    from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile


_WATER = LIE_CODES["water"]
_OB = LIE_CODES["ob"]
_TREES = LIE_CODES["trees"]
_ROUGH = LIE_CODES["rough"]
_SAND = LIE_CODES["sand"]
_HOLED = LIE_CODES["holed"]


@dataclass(frozen=True)
class TransitionOutcome:
    """One sampled outcome: next state and any added stroke penalty."""

    next_state: State
    extra_strokes: int


def _condition_modifiers(
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


def sample_transitions(
    state: State,
    action: ShotAction,
    profile: PlayerProfile,
    baseline: BaselineBag,
    conditions: CourseConditions,
    raster: LieRaster,
    n_samples: int,
    rng: np.random.Generator,
) -> list[TransitionOutcome]:
    """Sample ``n_samples`` next-state outcomes."""
    require(n_samples > 0, "n_samples must be > 0", n_samples)
    require(state.lie != _HOLED, "cannot act from holed state")

    dist_mult, sl_mult, slat_mult = _condition_modifiers(state.lie, conditions)
    dist = profile.effective_distance(action.club, baseline) * dist_mult
    distr = profile.effective_distribution(action.club, baseline).scaled(
        sl_mult, slat_mult
    )
    offsets = distr.sample(n_samples, rng)  # (n, 2): along, lateral

    # Aim-frame landing = (carry_mean + dlong, dlat). Rotate into hole frame.
    along = dist + offsets[:, 0]
    lateral = offsets[:, 1]
    cos_a = np.cos(action.aim_angle_rad)
    sin_a = np.sin(action.aim_angle_rad)
    dx = along * cos_a - lateral * sin_a
    dy = along * sin_a + lateral * cos_a

    xs = state.x + dx
    ys = state.y + dy

    outcomes: list[TransitionOutcome] = []
    for x, y in zip(xs, ys, strict=True):
        lie = raster.lie_at(float(x), float(y))
        outcomes.append(_apply_hazard_rules(state, float(x), float(y), lie))
    return outcomes


def _apply_hazard_rules(
    origin: State, x: float, y: float, landing_lie: int
) -> TransitionOutcome:
    """Standard USGA drop semantics (Phase 1): water/OB → drop at origin + 1 stroke."""
    if landing_lie in (_WATER, _OB):
        return TransitionOutcome(
            next_state=State(x=origin.x, y=origin.y, lie=origin.lie),
            extra_strokes=1,
        )
    return TransitionOutcome(
        next_state=State(x=x, y=y, lie=landing_lie),
        extra_strokes=0,
    )


# Suppress unused-import lint when typing block is dropped at runtime.
_NP_TYPING_HINT: NDArray[np.float64] | None = None
