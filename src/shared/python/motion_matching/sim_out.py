"""Engine-agnostic ``SimOut`` and ``SimFitResult`` dataclasses.

These mirror the MATLAB structures returned by ``simulate_with_coefficients.m``
and the cost evaluator. Every engine (Simscape, Drake, Pinocchio, MuJoCo, ...)
produces a :class:`SimOut` so the shared cost / leaderboard / plotting code
operates on a single canonical shape.

This module deliberately re-exports :class:`SimOutput` from :mod:`cost`
under the name :class:`SimOut` to align with the MATLAB nomenclature while
preserving backwards compatibility with the existing Python tests.

Public API:
    SimOut     -- engine-agnostic per-frame simulation output.
    SimFitResult -- the (theta, cost-breakdown, target, sim) tuple emitted by a
                    fit run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .cost import CostBreakdown, SimOutput
from .target import ClubTarget

# Aligning naming with MATLAB ``sim_out`` while keeping the existing class.
SimOut = SimOutput

__all__ = ["FitResult", "SimFitResult", "SimOut"]


@dataclass(frozen=True)
class SimFitResult:
    """Result of a single fit / inverse run.

    Attributes:
        theta:   Coefficient vector that minimised the cost (1-D float array).
        cost:    Final scalar cost ``J``.
        breakdown: Per-term :class:`CostBreakdown` for ``theta``.
        target:  The :class:`ClubTarget` that was fit to.
        sim:     The :class:`SimOut` produced by ``theta``.
        engine:  Free-form engine identifier, e.g. ``"simscape"``,
                 ``"drake"``, ``"mujoco"``, ``"pinocchio"``.
        n_iter:  Number of optimiser iterations consumed (0 if N/A).
        wallclock_s: Wallclock seconds for the run (0.0 if N/A).
    """

    theta: NDArray[np.float64]
    cost: float
    breakdown: CostBreakdown
    target: ClubTarget
    sim: SimOut
    engine: str
    n_iter: int = 0
    wallclock_s: float = 0.0

    def __post_init__(self) -> None:
        """Lightweight invariants -- cost finite, breakdown sums to cost."""
        if not np.isfinite(self.cost):
            raise ValueError(f"SimFitResult.cost must be finite; got {self.cost!r}")
        if self.cost < 0.0:
            raise ValueError(
                f"SimFitResult.cost must be non-negative; got {self.cost!r}"
            )
        # Allow a tiny float tolerance (the breakdown is computed in float64
        # by ``compute_cost`` so a few ulps of drift is normal).
        if abs(self.breakdown.total - self.cost) > 1e-9 * max(1.0, abs(self.cost)):
            raise ValueError(
                "FitResult.cost must equal breakdown.total "
                f"(got {self.cost!r} vs {self.breakdown.total!r})"
            )
        if not isinstance(self.engine, str) or not self.engine:
            raise ValueError("SimFitResult.engine must be a non-empty string")


FitResult = SimFitResult
