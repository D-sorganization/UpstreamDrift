"""Choose aim and pace by strokes gained or, for must-make putts, by make rate.

Every candidate action at one distance is simulated with the same seed
(common random numbers), so comparisons between actions are not swamped by
sampling noise. Decisions use the player's own curve; :func:`strokes_gained`
reports SG against that curve and against a reference field, side by side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
    PuttOutcomeModel,
    PuttOutcomes,
    continuation_row,
    curve_from_transitions,
    distance_seed,
    validated_grid,
)

OBJECTIVES = ("sg", "must_make")
_IMPROVEMENT_EPS = 1e-12
_MAX_POLICY_ITERATIONS = 50


@dataclass(frozen=True)
class ActionEvaluation:
    """Make rate and expected strokes of one action, with Monte Carlo errors."""

    action: PuttAction
    make_rate: float
    make_rate_se: float
    expected_strokes: float
    expected_strokes_se: float


def _strokes(outcomes: PuttOutcomes, baseline: ExpectedPuttsCurve) -> np.ndarray:
    """Per-sample strokes to hole out: 1, plus ``baseline``(leave) after a miss."""
    return 1.0 + np.where(outcomes.holed, 0.0, baseline(outcomes.leave_ft))


def _mean_se(samples: np.ndarray) -> tuple[float, float]:
    spread = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    return float(samples.mean()), spread / float(np.sqrt(samples.size))


def evaluate_outcomes(
    action: PuttAction, outcomes: PuttOutcomes, baseline: ExpectedPuttsCurve
) -> ActionEvaluation:
    """Summarise ``outcomes``; a miss costs 1 + ``baseline``(leave) strokes."""
    n, p = outcomes.holed.size, outcomes.make_rate
    mean, se = _mean_se(_strokes(outcomes, baseline))
    return ActionEvaluation(action, p, float(np.sqrt(p * (1 - p) / n)), mean, se)


def evaluate_actions(
    model: PuttOutcomeModel,
    distance_ft: float,
    actions: Sequence[PuttAction],
    baseline: ExpectedPuttsCurve,
    n: int,
    seed: int,
) -> list[ActionEvaluation]:
    """Evaluate every candidate from ``distance_ft`` on common random numbers."""
    require(len(actions) > 0, "need at least one candidate action")
    return [
        evaluate_outcomes(a, model.simulate(distance_ft, a, n, seed), baseline)
        for a in actions
    ]


def best_action(
    evaluations: Sequence[ActionEvaluation], objective: str = "sg"
) -> ActionEvaluation:
    """Fewest expected strokes (``"sg"``) or highest make rate (``"must_make"``).

    Must-make ignores the comeback putt except as a tie-break.
    """
    require(objective in OBJECTIVES, f"objective must be one of {OBJECTIVES}")
    if objective == "sg":
        return min(evaluations, key=lambda e: (e.expected_strokes, -e.make_rate))
    return min(evaluations, key=lambda e: (-e.make_rate, e.expected_strokes))


@dataclass(frozen=True)
class StrokesGained:
    """Mean strokes gained per putt, with Monte Carlo standard errors.

    ``sg_player`` is against the player's own E(d), so their best strategy is 0;
    ``sg_tour`` is against a reference field's E(d), as published stats quote.
    """

    sg_player: float
    sg_player_se: float
    sg_tour: float
    sg_tour_se: float


def strokes_gained(
    distance_ft: float,
    outcomes: PuttOutcomes,
    player: ExpectedPuttsCurve,
    tour: ExpectedPuttsCurve,
) -> StrokesGained:
    """SG = E(start) - 1 - E(leave) per putt, one curve for both ends of each value.

    Mixing them (tour E at the start, player E for the comeback) measures neither.
    """

    def against(curve: ExpectedPuttsCurve) -> tuple[float, float]:
        return _mean_se(float(curve(distance_ft)) - _strokes(outcomes, curve))

    return StrokesGained(*against(player), *against(tour))


def optimal_expected_putts(
    model: PuttOutcomeModel,
    distances_ft: Sequence[float] | np.ndarray,
    actions: Sequence[PuttAction],
    n: int,
    seed: int,
) -> tuple[ExpectedPuttsCurve, tuple[PuttAction, ...]]:
    """Player-optimal E(d) and the best action per grid distance (policy iteration).

    Starts from the most-holing action, then alternates an exact solve with a
    greedy improvement until no action improves by more than round-off.
    """
    grid = validated_grid(distances_ft)
    require(len(actions) > 0, "need at least one candidate action")
    rows = np.empty((grid.size, len(actions), grid.size))
    make = np.empty((grid.size, len(actions)))
    for i, distance in enumerate(grid):
        seed_i = distance_seed(seed, i)
        for k, action in enumerate(actions):
            outcomes = model.simulate(float(distance), action, n, seed_i)
            rows[i, k], make[i, k] = (
                continuation_row(grid, outcomes),
                outcomes.make_rate,
            )
    choice = make.argmax(axis=1)
    for _ in range(_MAX_POLICY_ITERATIONS):
        curve = curve_from_transitions(grid, rows[np.arange(grid.size), choice])
        q = 1.0 + rows @ curve.values
        current = q[np.arange(grid.size), choice]
        better = q.min(axis=1) < current - _IMPROVEMENT_EPS
        if not better.any():
            break
        choice = np.where(better, q.argmin(axis=1), choice)
    return curve, tuple(actions[k] for k in choice)
