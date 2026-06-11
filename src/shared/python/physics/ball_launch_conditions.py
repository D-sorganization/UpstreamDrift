"""Launch conditions, environmental conditions, and trajectory data structures.

This submodule contains dataclasses representing the inputs and outputs of
ball flight simulation, extracted from ball_flight_physics.py as part of P1
sprint decomposition (issue #2486).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.shared.python.core.constants import AIR_DENSITY_SEA_LEVEL_KG_M3, GRAVITY_M_S2
from src.shared.python.physics.atmosphere import air_density as _isa_air_density


@dataclass(frozen=True)
class LaunchConditions:
    """Initial launch conditions for the ball-flight simulator.

    Units are part of the public simulator contract:

    * ``velocity`` is ball speed in metres per second.
    * ``launch_angle`` is vertical launch angle in radians above horizontal.
    * ``azimuth_angle`` is horizontal bearing in radians, with 0 along +x.
    * ``spin_rate`` is backspin magnitude in RPM.
    * ``spin_axis`` is the spin-axis unit vector.

    User-facing golf inputs are normally degrees and RPM. Build those with
    :meth:`from_user_units` so degree-to-radian conversion stays in one place.
    """

    velocity: float
    launch_angle: float
    azimuth_angle: float = 0.0
    spin_rate: float = 0.0
    spin_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))

    @classmethod
    def from_user_units(
        cls,
        velocity: float,
        launch_angle_deg: float,
        azimuth_deg: float = 0.0,
        spin_rate_rpm: float = 0.0,
        spin_axis: np.ndarray | None = None,
    ) -> LaunchConditions:
        """Build launch conditions from golf-facing units.

        ``launch_angle_deg`` and ``azimuth_deg`` are converted to radians.
        ``spin_rate_rpm`` is already the stored unit and is passed through.
        """
        if not math.isfinite(velocity) or velocity < 0.0:
            raise ValueError(f"velocity must be finite and >= 0, got {velocity!r}")
        if not math.isfinite(spin_rate_rpm) or spin_rate_rpm < 0.0:
            raise ValueError(
                f"spin_rate_rpm must be finite and >= 0, got {spin_rate_rpm!r}"
            )
        axis = (
            np.asarray(spin_axis, dtype=float)
            if spin_axis is not None
            else np.array([0.0, -1.0, 0.0])
        )
        return cls(
            velocity=float(velocity),
            launch_angle=math.radians(launch_angle_deg),
            azimuth_angle=math.radians(azimuth_deg),
            spin_rate=float(spin_rate_rpm),
            spin_axis=axis,
        )


@dataclass(frozen=True)
class EnvironmentalConditions:
    """Environmental settings.

    ``air_density`` is the explicit value used by the integrator. Callers
    that wish to derive it from ``altitude`` and ``temperature`` should use
    :meth:`from_altitude` (recommended) which evaluates the ISA-troposphere
    model in :func:`src.shared.python.physics.atmosphere.air_density`.

    See issue #3504 for the environmental-gradient feature this supports.
    """

    air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3)
    wind_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    gravity: float = float(GRAVITY_M_S2)
    altitude: float = 0.0
    temperature: float = 15.0
    sea_level_pressure_pa: float | None = None

    @classmethod
    def from_altitude(
        cls,
        altitude_m: float = 0.0,
        temperature_c: float = 15.0,
        wind_velocity: np.ndarray | None = None,
        gravity: float = float(GRAVITY_M_S2),
        pressure_pa: float | None = None,
    ) -> EnvironmentalConditions:
        """Build an ``EnvironmentalConditions`` with ISA-derived density.

        Parameters
        ----------
        altitude_m
            Altitude above mean sea level [m]. Validated by
            :func:`src.shared.python.physics.atmosphere.air_density`.
        temperature_c
            Ground (sea-level reference) temperature [C]. Validated by the
            same helper.
        wind_velocity
            Wind velocity [m/s]; defaults to zero.
        gravity
            Gravitational acceleration magnitude [m/s^2].
        pressure_pa
            Optional sea-level pressure override [Pa]; defaults to 101325.

        Returns
        -------
        EnvironmentalConditions
            Frozen dataclass with ``air_density`` populated from ISA.
        """
        rho = _isa_air_density(
            altitude_m=altitude_m,
            temperature_c=temperature_c,
            pressure_pa=pressure_pa,
        )
        wind = (
            np.asarray(wind_velocity, dtype=float)
            if wind_velocity is not None
            else np.array([0.0, 0.0, 0.0])
        )
        return cls(
            air_density=rho,
            wind_velocity=wind,
            gravity=float(gravity),
            altitude=float(altitude_m),
            temperature=float(temperature_c),
            sea_level_pressure_pa=(
                float(pressure_pa) if pressure_pa is not None else None
            ),
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
