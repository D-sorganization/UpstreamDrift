"""Fixed-pivot and prescribed moving-hub conditions (TB-03 #10588).

Distinguishes fixed-pivot models from moving-hub variants and tracks external
work and power contributions, preventing externally driven hub motion from being
silently mislabeled as an unforced or free baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixedPivotHub:
    """Fixed shoulder/pivot anchor condition."""

    pivot_world: np.ndarray
    is_moving: bool = False
    external_work_joules: float = 0.0


@dataclass(frozen=True)
class MovingHub:
    """Prescribed moving hub with external kinematics and work tracking."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    is_moving: bool = True
    external_work_joules: float = 0.0

    @classmethod
    def from_trajectory(
        cls,
        times: np.ndarray,
        positions: np.ndarray,
        forces: np.ndarray | None = None,
    ) -> MovingHub:
        """Construct a moving hub instance and integrate external power."""
        t = np.asarray(times, dtype=float)
        pos = np.asarray(positions, dtype=float)
        n = len(t)
        if pos.shape[0] != n or pos.shape[1] != 3:
            raise ValueError(f"Positions shape must be ({n}, 3), got {pos.shape}")

        # Compute velocities by gradient
        vel = np.gradient(pos, t, axis=0)

        f = np.zeros_like(pos) if forces is None else np.asarray(forces, dtype=float)
        if f.shape != pos.shape:
            raise ValueError(f"Forces shape must match positions, got {f.shape}")

        # Power = F . v (instantaneous)
        power = np.sum(f * vel, axis=-1)
        # Numerical integration using trapezoid rule
        work = float(np.trapezoid(power, t))

        return cls(
            times=t,
            positions=pos,
            velocities=vel,
            forces=f,
            is_moving=True,
            external_work_joules=work,
        )
