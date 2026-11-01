"""Plan on the green the player perceives, execute on the true green (#14173).

A misread is not just noise on the start line. A player who under-reads slope
picks a wrong aim *and* a wrong pace, and both errors grow with distance,
break and speed. So the player's action is the optimum (strokes gained or
must-make) on the *perceived* surface under their own stroke noise, and that
action is then executed, with the same noise model, on the *true* surface.

A :class:`GreenRead` transforms the true surface:

* ``slope_scale`` multiplies every height, so every gradient (slope magnitude
  and therefore break) scales by it; 0.5 is a 50 % under-read.
* ``stimp_ratio`` is perceived over true green speed; rolling resistance goes
  as 1/stimp, so the perceived ``mu_roll0`` is the true one divided by it.

Plans are cached on a grid of reads (``read_steps_per_unit``) so sampling many
reads from a skill profile costs one optimisation per distinct grid cell. Every
plan is scored on the same planning seed (common random numbers), and
execution draws its own noise from the seed passed to :meth:`play`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.putting_analytics.expected_putts import (
    ExpectedPuttsCurve,
    PuttAction,
    PuttOutcomes,
    distance_seed,
)
from src.shared.python.putting_analytics.optimize import (
    OBJECTIVES,
    ActionEvaluation,
    best_action,
    evaluate_outcomes,
)
from src.shared.python.putting_analytics.roll_model import (
    BatchRollModel,
    ExecutionNoise,
    perfect_execution,
)
from src.shared.python.putting_dynamics.surfaces import SurfaceSpec


@dataclass(frozen=True)
class GreenRead:
    """How the player perceives the green relative to the truth (1.0 = exact)."""

    slope_scale: float = 1.0
    stimp_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name in ("slope_scale", "stimp_ratio"):
            value = getattr(self, name)
            require(math.isfinite(value) and value > 0.0, f"{name} must be > 0", value)

    @property
    def is_perfect(self) -> bool:
        """True when the read equals the true green."""
        return self.slope_scale == 1.0 and self.stimp_ratio == 1.0

    def snapped(self, steps_per_unit: int) -> GreenRead:
        """Nearest read on the cache grid; a perfect read stays exactly perfect."""
        require(steps_per_unit > 0, "steps_per_unit must be positive", steps_per_unit)
        return GreenRead(
            round(self.slope_scale * steps_per_unit) / steps_per_unit,
            round(self.stimp_ratio * steps_per_unit) / steps_per_unit,
        )


def perceive(surface: SurfaceSpec, read: GreenRead) -> SurfaceSpec:
    """The green as the player reads it; the true surface for a perfect read."""
    if read.is_perfect:
        return surface
    height = dataclasses.replace(
        surface.height, heights_m=surface.height.heights_m * read.slope_scale
    )
    friction = dataclasses.replace(
        surface.friction, mu_roll0=surface.friction.mu_roll0 / read.stimp_ratio
    )
    return dataclasses.replace(surface, height=height, friction=friction)


@dataclass
class PerceivedGreenPlanner:
    """Chooses actions on perceived greens and plays them on the true green.

    ``baseline`` is the curve that prices a miss (the player's own E(d) for
    their strategy); ``candidates`` is the action set searched on each plan;
    ``plan_seed`` fixes the noise every plan is scored on.
    """

    surface: SurfaceSpec
    candidates: tuple[PuttAction, ...]
    baseline: ExpectedPuttsCurve
    n: int
    execution: ExecutionNoise = perfect_execution
    objective: str = "sg"
    plan_seed: int = 0
    read_steps_per_unit: int = 20
    _plans: dict[tuple[float, GreenRead], ActionEvaluation] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        require(len(self.candidates) > 0, "need at least one candidate action")
        require(self.n > 1, "need at least two putts per evaluation", self.n)
        require(self.objective in OBJECTIVES, f"objective must be in {OBJECTIVES}")

    @property
    def true_model(self) -> BatchRollModel:
        """Roll model on the true green with the player's stroke noise."""
        return BatchRollModel(self.surface, self.execution)

    def plan(self, distance_ft: float, read: GreenRead) -> ActionEvaluation:
        """Best action on the perceived green (cached per read grid cell)."""
        cell = read if read.is_perfect else read.snapped(self.read_steps_per_unit)
        key = (distance_ft, cell)
        if key not in self._plans:
            model = BatchRollModel(perceive(self.surface, cell), self.execution)
            outcomes = model.simulate_many(
                distance_ft, self.candidates, self.n, self.plan_seed
            )
            evaluations = [
                evaluate_outcomes(action, sample, self.baseline)
                for action, sample in zip(self.candidates, outcomes, strict=True)
            ]
            self._plans[key] = best_action(evaluations, self.objective)
        return self._plans[key]

    def play(self, distance_ft: float, read: GreenRead, seed: int) -> PuttOutcomes:
        """Execute the perceived-green plan ``n`` times on the true green."""
        action = self.plan(distance_ft, read).action
        return self.true_model.simulate(distance_ft, action, self.n, seed)


@dataclass(frozen=True)
class ReadErrorCost:
    """Expected strokes with the sampled reads, with a perfect read, and the gap.

    ``cost`` is paired putt by putt on common execution noise, so its standard
    error reflects the read errors, not stroke noise.
    """

    expected_strokes: float
    perfect_read_strokes: float
    cost: float
    cost_se: float


def _strokes(outcomes: PuttOutcomes, baseline: ExpectedPuttsCurve) -> np.ndarray:
    return 1.0 + np.where(outcomes.holed, 0.0, baseline(outcomes.leave_ft))


def read_error_cost(
    planner: PerceivedGreenPlanner,
    distance_ft: float,
    reads: Sequence[GreenRead],
    seed: int,
) -> ReadErrorCost:
    """Expected-strokes cost of misreading this putt, over sampled ``reads``.

    Each read gets its own plan and its own seed stream; the perfect-read
    comparison reuses the same seed per read, so execution noise is shared.
    """
    require(len(reads) > 0, "need at least one sampled read")
    misread, perfect = [], []
    for index, read in enumerate(reads):
        seed_i = distance_seed(seed, index)
        misread.append(
            _strokes(planner.play(distance_ft, read, seed_i), planner.baseline)
        )
        perfect.append(
            _strokes(planner.play(distance_ft, GreenRead(), seed_i), planner.baseline)
        )
    with_reads, exact = np.concatenate(misread), np.concatenate(perfect)
    gap = with_reads - exact
    return ReadErrorCost(
        float(with_reads.mean()),
        float(exact.mean()),
        float(gap.mean()),
        float(gap.std(ddof=1) / math.sqrt(gap.size)),
    )
