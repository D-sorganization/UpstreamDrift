"""Personalised vs tour-baseline strategy on one hole (SHOW-B4, #14568)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.sg_optimizer.course.conditions import CourseConditions
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES, LieRaster
from src.shared.python.sg_optimizer.mdp.action import ActionSet, ShotAction
from src.shared.python.sg_optimizer.mdp.state import State
from src.shared.python.sg_optimizer.mdp.value_iteration import HoleMDP, SolveResult
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


@dataclass(frozen=True)
class StrategyProblem:
    """The hole and player-independent inputs shared by both solves."""

    raster: LieRaster
    baseline: BaselineBag
    conditions: CourseConditions
    actions: ActionSet
    start: State


@dataclass(frozen=True)
class SolveSettings:
    """Sampling and iteration controls (same seed for tour and player)."""

    n_samples: int = 64
    seed: int = 0
    max_iter: int = 200


def green_terminal(raster: LieRaster) -> LieRaster:
    """Copy of ``raster`` where every green cell ends the hole.

    The MDP has no putting model, so with only the pin cell terminal the
    expected strokes do not converge. Ending on the green gives *strokes to
    reach the green*, which converges and still ranks tee shots by their
    hazard and dispersion risk; add putting separately.
    """
    codes = raster.codes.copy()
    codes[codes == LIE_CODES["green"]] = LIE_CODES["holed"]
    return LieRaster(codes, raster.resolution_yd, raster.origin, raster.pin, raster.tee)


def solve_mdp(
    problem: StrategyProblem,
    profile: PlayerProfile,
    model: ShotOutcomeModel | None,
    cfg: SolveSettings,
) -> tuple[HoleMDP, SolveResult]:
    """Solve the hole for ``profile``; return the MDP and its solve result."""
    mdp = HoleMDP(
        problem.raster,
        profile,
        problem.baseline,
        problem.conditions,
        problem.actions,
        cfg.n_samples,
        cfg.seed,
        model=model,
    )
    return mdp, mdp.solve(max_iter=cfg.max_iter)


def _solve(
    problem: StrategyProblem,
    profile: PlayerProfile,
    model: ShotOutcomeModel | None,
    cfg: SolveSettings,
) -> tuple[float, ShotAction]:
    mdp, result = solve_mdp(problem, profile, model, cfg)
    return (
        mdp.expected_strokes(problem.start, result.value),
        mdp.optimal_action(problem.start, result.value),
    )


def compare_strategies(
    problem: StrategyProblem,
    profile: PlayerProfile,
    *,
    model: ShotOutcomeModel | None = None,
    settings: SolveSettings = SolveSettings(),  # noqa: B008 - frozen, immutable
) -> StrategyReport:
    """Solve the hole for the tour baseline and for ``profile`` (same seed)."""
    tour = PlayerProfile(name="tour", baseline=profile.baseline)
    t, ta = _solve(problem, tour, None, settings)
    p, pa = _solve(problem, profile, model, settings)
    return StrategyReport(t, p, ta, pa)
