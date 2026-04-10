from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.engines.physics_engines.putting_green.python.ball_roll_physics import RollMode


@dataclass
class SimulationConfig:
    """Configuration for putting simulation.

    Attributes:
        timestep: Simulation time step [s]
        max_simulation_time: Maximum simulation duration [s]
        stopping_velocity_threshold: Speed below which ball stops [m/s]
        record_trajectory: Whether to record full trajectory
        integrator: Integration method ("euler", "rk4", "verlet")
    """

    timestep: float = 0.001
    max_simulation_time: float = 30.0
    stopping_velocity_threshold: float = 0.005
    record_trajectory: bool = True
    integrator: str = "euler"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.timestep <= 0:
            raise ValueError(f"timestep must be positive, got {self.timestep}")
        if self.max_simulation_time <= 0:
            raise ValueError(
                f"max_simulation_time must be positive, got {self.max_simulation_time}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestep": self.timestep,
            "max_simulation_time": self.max_simulation_time,
            "stopping_velocity_threshold": self.stopping_velocity_threshold,
            "record_trajectory": self.record_trajectory,
            "integrator": self.integrator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class SimulationResult:
    """Result of a putting simulation.

    Attributes:
        positions: Array of ball positions [[x, y], ...]
        velocities: Array of ball velocities [[vx, vy], ...]
        times: Array of time stamps [t0, t1, ...]
        holed: Whether ball went in hole
        final_position: Final ball position
        spins: Optional array of spin vectors
        modes: Optional list of roll modes at each step
    """

    positions: np.ndarray
    velocities: np.ndarray
    times: np.ndarray
    holed: bool
    final_position: np.ndarray
    spins: np.ndarray | None = None
    modes: list[RollMode] | None = None

    @property
    def total_distance(self) -> float:
        """Compute total distance rolled."""
        if len(self.positions) < 2:
            return 0.0

        distances = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        return float(np.sum(distances))

    @property
    def duration(self) -> float:
        """Total simulation duration."""
        return float(self.times[-1] - self.times[0])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "times": self.times.tolist(),
            "holed": self.holed,
            "final_position": self.final_position.tolist(),
            "total_distance": self.total_distance,
            "duration": self.duration,
        }
