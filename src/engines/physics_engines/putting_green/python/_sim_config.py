from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    ROLL_MODEL_FIELD,
    UD_LEGACY_ROLL_MODEL,
    RollMode,
    require_roll_model,
    validate_roll_model_name,
)


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
        roll_model: Name of the roll model that produced this result
            (ADR-0045 F1). Always present in :meth:`to_dict`; results from
            different models must never be compared without it.
    """

    positions: np.ndarray
    velocities: np.ndarray
    times: np.ndarray
    holed: bool
    final_position: np.ndarray
    spins: np.ndarray | None = None
    modes: list[RollMode] | None = None
    roll_model: str = field(default=UD_LEGACY_ROLL_MODEL)

    def __post_init__(self) -> None:
        """Refuse a result that cannot name the physics that produced it.

        Raises:
            RollModelProvenanceError: If ``roll_model`` is blank or unknown.
        """
        validate_roll_model_name(self.roll_model, source="SimulationResult")

    @property
    def total_distance(self) -> float:
        """Compute total distance rolled."""
        if len(self.positions) < 2:
            return 0.0

        diffs = np.diff(self.positions, axis=0)
        distances = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        return float(np.sum(distances))

    @property
    def duration(self) -> float:
        """Total simulation duration."""
        return float(self.times[-1] - self.times[0])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a result document that names its roll model.

        Postcondition: the document always carries ``roll_model`` (ADR-0045).
        """
        return {
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "times": self.times.tolist(),
            "holed": self.holed,
            "final_position": self.final_position.tolist(),
            "total_distance": self.total_distance,
            "duration": self.duration,
            ROLL_MODEL_FIELD: self.roll_model,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        source: str = "putting-green result document",
    ) -> SimulationResult:
        """Deserialize a result document, refusing an unnamed payload.

        This is a fail-closed reader (ADR-0045 F1): a document without a
        ``roll_model`` field cannot be interpreted, because the two preserved
        roll models differ by the ~2.854 roll-out ratio (Tools#4819).

        Args:
            data: Result document previously produced by :meth:`to_dict`.
            source: Human-readable origin quoted in error messages.

        Returns:
            The reconstructed result, tagged with the document's roll model.

        Raises:
            RollModelProvenanceError: If the document does not name a
                preserved roll model.
            KeyError: If a required trajectory field is missing.
        """
        roll_model = require_roll_model(data, source=source)
        return cls(
            positions=np.asarray(data["positions"], dtype=float).reshape(-1, 2),
            velocities=np.asarray(data["velocities"], dtype=float).reshape(-1, 2),
            times=np.asarray(data["times"], dtype=float).reshape(-1),
            holed=bool(data["holed"]),
            final_position=np.asarray(data["final_position"], dtype=float).reshape(-1),
            roll_model=roll_model,
        )
