"""Personalised vs tour-baseline strategy on one hole (SHOW-B4, #14568)."""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LieRaster
from src.shared.python.sg_optimizer.mdp.action import ActionSet, ShotAction
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.mdp.value_iteration import HoleMDP
from src.shared.python.sg_optimizer.shot_model.baseline import BaselineBag
from src.shared.python.sg_optimizer.shot_model.outcome import ShotOutcomeModel
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile


@dataclass(frozen=True)
class StrategyReport:
    """Expected strokes and first shot for the tour baseline and for the player."""

    tour_strokes: float
    personal_strokes: float
    tour_action: ShotAction
    personal_action: ShotAction

    @property
    def delta(self) -> float:
        """Personalised minus tour expected strokes (positive = worse than tour)."""
        return self.personal_strokes - self.tour_strokes


def _solve(
    raster: LieRaster,
    profile: PlayerProfile,
    baseline: BaselineBag,
    conditions: CourseConditions,
    actions: ActionSet,
    start: State,
    model: ShotOutcomeModel | None,
    n_samples: int,
    seed: int,
    max_iter: int,
) -> tuple[float, ShotAction]:
    mdp = HoleMDP(
        raster, profile, baseline, conditions, actions, n_samples, seed, model=model
    )
    value = mdp.solve(max_iter=max_iter).value
    return mdp.expected_strokes(start, value), mdp.optimal_action(start, value)


def compare_strategies(
    raster: LieRaster,
    profile: PlayerProfile,
    baseline: BaselineBag,
    conditions: CourseConditions,
    actions: ActionSet,
    start: State,
    *,
    model: ShotOutcomeModel | None = None,
    n_samples: int = 64,
    seed: int = 0,
    max_iter: int = 200,
) -> StrategyReport:
    """Solve the hole for the tour baseline and for ``profile`` (same seed)."""
    tour = PlayerProfile(name="tour", baseline=profile.baseline)
    args = (raster, baseline, conditions, actions, start)
    t, ta = _solve(args[0], tour, *args[1:], None, n_samples, seed, max_iter)
    p, pa = _solve(args[0], profile, *args[1:], model, n_samples, seed, max_iter)
    return StrategyReport(t, p, ta, pa)
