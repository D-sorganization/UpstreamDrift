"""A player's own expected-putts curve E_player(d), solved exactly.

For grid distances d_i, E_i = 1 + mean over missed putts of E(leave), where
E(leave) is linear interpolation on the grid. That is a linear system
(I - M) E = 1, solved directly rather than iterated. Using every leave
sample, not the mean leave, avoids Jensen bias because E is convex.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.shared.python.contracts import require

_VALUE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PuttAction:
    """Plan for one putt: start-line aim offset (deg) and target pace (ft past).

    The outcome model owns the sign convention of ``aim_offset_deg``.
    """

    aim_offset_deg: float
    pace_ft_past: float


@dataclass(frozen=True)
class PuttOutcomes:
    """Monte Carlo result of one action: holed flags and leave distances (ft)."""

    holed: np.ndarray
    leave_ft: np.ndarray

    def __post_init__(self) -> None:
        require(self.holed.ndim == 1 and self.holed.size > 0, "need a 1-D sample")
        require(self.holed.shape == self.leave_ft.shape, "holed/leave shape mismatch")
        require(bool(np.all(self.leave_ft >= 0.0)), "leave_ft must be non-negative")

    @property
    def make_rate(self) -> float:
        """Fraction of the sample holed."""
        return float(np.mean(self.holed))


class PuttOutcomeModel(Protocol):
    """Anything that can simulate ``n`` putts of one action from a distance."""

    def simulate(
        self, distance_ft: float, action: PuttAction, n: int, seed: int
    ) -> PuttOutcomes:
        """Return ``n`` outcomes; the same seed must reproduce the same outcomes."""
        ...


@dataclass(frozen=True)
class ExpectedPuttsCurve:
    """E(d) on a grid; interpolates inside and clamps outside the grid."""

    distances_ft: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        require(self.distances_ft.shape == self.values.shape, "shape mismatch")
        require(bool(np.all(np.diff(self.distances_ft) > 0)), "grid must increase")
        require(bool(np.all(self.values >= 1.0 - _VALUE_TOLERANCE)), "E(d) < 1")

    def __call__(self, distance_ft: float | np.ndarray) -> np.ndarray:
        """Expected putts to hole out from ``distance_ft``."""
        return np.interp(distance_ft, self.distances_ft, self.values)


def distance_seed(seed: int, index: int) -> int:
    """Seed for grid point ``index``; shared by every action there (common numbers)."""
    return int(np.random.SeedSequence(seed, spawn_key=(index,)).generate_state(1)[0])


def validated_grid(distances_ft: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the grid as an array after checking it is usable."""
    grid = np.asarray(distances_ft, dtype=float)
    require(grid.ndim == 1 and grid.size >= 2, "grid needs at least two distances")
    require(bool(np.all(np.diff(grid) > 0)) and grid[0] > 0, "grid must increase")
    return grid


def continuation_row(grid: np.ndarray, outcomes: PuttOutcomes) -> np.ndarray:
    """Row of M: interpolation weights of missed leaves, averaged over the sample.

    Leaves beyond the grid are clamped to its ends, so make the grid reach well
    past the longest leave you expect.
    """
    leave = np.clip(outcomes.leave_ft[~outcomes.holed], grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, leave, side="right"), 1, grid.size - 1)
    frac = (leave - grid[hi - 1]) / (grid[hi] - grid[hi - 1])
    row = np.zeros(grid.size)
    np.add.at(row, hi - 1, 1.0 - frac)
    np.add.at(row, hi, frac)
    return row / outcomes.holed.size


def curve_from_transitions(
    grid: np.ndarray, transition: np.ndarray
) -> ExpectedPuttsCurve:
    """Solve (I - M) E = 1 for the expected-putts values on ``grid``."""
    system = np.eye(grid.size) - transition
    if np.linalg.cond(system) > 1e12:
        raise ValueError("some distance is never holed out: E(d) is unbounded")
    return ExpectedPuttsCurve(grid, np.linalg.solve(system, np.ones(grid.size)))


def solve_expected_putts(
    model: PuttOutcomeModel,
    distances_ft: Sequence[float] | np.ndarray,
    policy: Callable[[float], PuttAction],
    n: int,
    seed: int,
) -> ExpectedPuttsCurve:
    """E(d) for a fixed habit ``policy`` (for example "always 1.5 ft past")."""
    grid = validated_grid(distances_ft)
    rows = [
        continuation_row(grid, model.simulate(float(d), policy(float(d)), n, s))
        for d, s in ((d, distance_seed(seed, i)) for i, d in enumerate(grid))
    ]
    return curve_from_transitions(grid, np.array(rows))
