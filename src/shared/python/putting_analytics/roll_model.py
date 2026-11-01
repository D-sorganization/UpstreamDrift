"""``PuttOutcomeModel`` backed by the batch roll integrator (#14171).

Frame: ball at the origin, hole at ``(d, 0)``; the surface's own slope and
grain set the break. ``aim_offset_deg`` turns the start line left (CCW, +y).
``pace_ft_past`` maps to a start speed through the constant-deceleration
roll law, using the base rolling resistance plus the along-line grade at the
ball: ``v0**2 = 2 g (mu_r0 + dh/dx) (d + pace)``. That is exact for a pure
roll on a uniform plane with no grain or speed dependence, and a monotone
nominal otherwise; the optimiser searches pace, so it needs no more.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.putting_analytics.batch_roll import (
    BatchRollResult,
    Capture,
    roll_batch,
)
from src.shared.python.putting_analytics.expected_putts import (
    PuttAction,
    PuttOutcomes,
)
from src.shared.python.putting_dynamics.surfaces import SurfaceSpec

FT_M = 0.3048

#: ``(rng, n) -> (start-line error [deg], speed ratio)`` per putt.
ExecutionNoise = Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]


def perfect_execution(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """No stroke error: every putt starts on line at the planned speed."""
    return np.zeros(n), np.ones(n)


@dataclass(frozen=True)
class BatchRollModel:
    """Simulate putts on ``surface`` with ``execution`` noise on each stroke."""

    surface: SurfaceSpec
    execution: ExecutionNoise = perfect_execution
    capture: Capture = "effective_radius"

    def nominal_speed_mps(self, distance_ft: float, pace_ft_past: float) -> float:
        """Start speed that rolls ``pace_ft_past`` beyond the hole (module law)."""
        decel = self.surface.friction.mu_roll0 + self.surface.height.gradient(0, 0)[0]
        require(decel > 0.0, "green too steep downhill for a nominal pace", decel)
        roll_m = (distance_ft + pace_ft_past) * FT_M
        require(roll_m > 0.0, "planned roll must be positive", roll_m)
        return float(np.sqrt(2.0 * GRAVITY_M_S2 * decel * roll_m))

    def roll_many(
        self, distance_ft: float, actions: Sequence[PuttAction], n: int, seed: int
    ) -> BatchRollResult:
        """``n`` putts of every action in one batch, action-major (rows k*n..).

        Every action shares the same execution noise draws (common random
        numbers), and each putt rolls exactly as it would alone.
        """
        require(distance_ft > 0.0 and n > 0, "need a positive distance and count")
        require(len(actions) > 0, "need at least one action")
        line_err, ratio = self.execution(np.random.default_rng(seed), n)
        aim = np.repeat([a.aim_offset_deg for a in actions], n)
        nominal = [self.nominal_speed_mps(distance_ft, a.pace_ft_past) for a in actions]
        heading = np.radians(aim + np.tile(line_err, len(actions)))
        speed = np.repeat(nominal, n) * np.tile(ratio, len(actions))
        zeros = np.zeros(speed.size)
        start = np.stack(
            (zeros, zeros, speed * np.cos(heading), speed * np.sin(heading), speed)
        )
        return roll_batch(start, self.surface, (distance_ft * FT_M, 0.0), self.capture)

    def roll(
        self, distance_ft: float, action: PuttAction, n: int, seed: int
    ) -> BatchRollResult:
        """Raw batch result (rest positions in metres, hole at ``(d, 0)``)."""
        return self.roll_many(distance_ft, (action,), n, seed)

    def simulate_many(
        self, distance_ft: float, actions: Sequence[PuttAction], n: int, seed: int
    ) -> list[PuttOutcomes]:
        """Outcomes of every action, rolled together on common noise."""
        result = self.roll_many(distance_ft, actions, n, seed)
        leave_m = np.hypot(result.rest_x_m - distance_ft * FT_M, result.rest_y_m)
        leave_ft = np.where(result.holed, 0.0, leave_m / FT_M)
        return [
            PuttOutcomes(
                result.holed[k * n : (k + 1) * n], leave_ft[k * n : (k + 1) * n]
            )
            for k in range(len(actions))
        ]

    def simulate(
        self, distance_ft: float, action: PuttAction, n: int, seed: int
    ) -> PuttOutcomes:
        """Roll ``n`` putts of ``action``; the same seed gives the same outcomes."""
        return self.simulate_many(distance_ft, (action,), n, seed)[0]
