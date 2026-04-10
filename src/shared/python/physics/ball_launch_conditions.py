"""Launch conditions, environmental conditions, and trajectory data structures.

This submodule contains dataclasses representing the inputs and outputs of
ball flight simulation, extracted from ball_flight_physics.py as part of P1
sprint decomposition (issue #2486).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.shared.python.core.constants import AIR_DENSITY_SEA_LEVEL_KG_M3, GRAVITY_M_S2


@dataclass(frozen=True)
class LaunchConditions:
    """Initial launch conditions."""

    velocity: float
    launch_angle: float
    azimuth_angle: float = 0.0
    spin_rate: float = 0.0
    spin_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))


@dataclass(frozen=True)
class EnvironmentalConditions:
    """Environmental settings."""

    air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3)
    wind_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    gravity: float = float(GRAVITY_M_S2)
    altitude: float = 0.0
    temperature: float = 15.0


@dataclass
class TrajectoryPoint:
    """Single point in trajectory."""

    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    forces: dict[str, np.ndarray]

    @property
    def speed(self) -> float:
        """Return the scalar speed from the velocity vector."""
        return float(np.linalg.norm(self.velocity))

    @property
    def height(self) -> float:
        """Return the vertical position component."""
        return float(self.position[2])
