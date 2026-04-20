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

    @classmethod
    def from_altitude(
        cls,
        altitude_m: float,
        wind_velocity: np.ndarray | None = None,
    ) -> EnvironmentalConditions:
        """Create conditions with ISA-derived air density for the given altitude.

        Uses the International Standard Atmosphere troposphere model
        (valid 0–11 km).  Air density decreases by ~1.2% per 100 m.

        Args:
            altitude_m: Altitude above sea level [m]. Must be ≥ 0.
            wind_velocity: 3-vector wind velocity [m/s]. Defaults to zero.

        Returns:
            EnvironmentalConditions with density and temperature set from ISA.

        Raises:
            ValueError: If altitude_m is negative.
        """
        if altitude_m < 0:
            raise ValueError("altitude_m must be non-negative")
        T0, P0 = 288.15, 101325.0  # K, Pa — ISA sea-level values
        L, g_isa, M_air, R_gas = 0.0065, 9.80665, 0.0289644, 8.31447
        T_k = T0 - L * altitude_m
        P = P0 * (T_k / T0) ** (g_isa * M_air / (R_gas * L))
        rho = P * M_air / (R_gas * T_k)
        return cls(
            air_density=rho,
            altitude=altitude_m,
            temperature=T_k - 273.15,
            wind_velocity=wind_velocity if wind_velocity is not None else np.zeros(3),
        )


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
