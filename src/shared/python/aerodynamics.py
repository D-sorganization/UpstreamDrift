"""Aerodynamics module for golf ball flight simulation.

This module provides sophisticated, tunable aerodynamic models that can be
toggled on/off for comparing trajectories with and without air resistance.

Design Principles (Pragmatic Programmer):
- Reversible: All effects can be toggled on/off at runtime
- Reusable: Modular components that compose well together
- DRY: Shared calculations extracted into helper functions
- Orthogonal: Independent components with no hidden coupling

Key Components:
- AerodynamicsConfig: Immutable configuration with toggles
- DragModel: Velocity-dependent drag with Reynolds correction
- LiftModel: Spin-induced lift (backspin effect)
- MagnusModel: Spin-induced lateral force (hook/slice)
- WindModel: Sophisticated wind with gusts and turbulence
- EnvironmentRandomizer: Stochastic environment simulation
- AerodynamicsEngine: Unified force calculation engine

References:
    - Bearman, P.W. & Harvey, J.K. (1976). Golf ball aerodynamics.
    - Smits, A.J. & Ogg, S. (2004). Golf ball aerodynamics. Physics Today.
    - Jorgensen, T. (1999). The Physics of Golf. Springer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.shared.python.physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    AIR_VISCOSITY_KG_M_S,
    GOLF_BALL_CROSS_SECTIONAL_AREA_M2,
    GOLF_BALL_DRAG_COEFFICIENT,
    GOLF_BALL_LIFT_COEFFICIENT,
    GOLF_BALL_RADIUS_M,
    MAGNUS_COEFFICIENT,
    SPIN_DECAY_RATE_S,
)

# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass(frozen=True)
class AerodynamicsConfig:
    """Immutable configuration for aerodynamic effects.

    All aerodynamic effects can be toggled independently (Orthogonal).
    The master `enabled` switch overrides individual toggles.

    Attributes:
        enabled: Master switch for all aerodynamic effects
        drag_enabled: Enable drag force (air resistance)
        lift_enabled: Enable lift force (from backspin)
        magnus_enabled: Enable Magnus force (from spin)
        drag_coefficient: Base drag coefficient (Cd)
        lift_coefficient: Base lift coefficient (Cl)
        magnus_coefficient: Magnus effect coefficient (Cm)
        spin_decay_rate: Rate of spin decay [1/s]
        reynolds_correction_enabled: Apply Reynolds number correction to Cd
        ball_radius: Ball radius for calculations [m]
        ball_area: Ball cross-sectional area [m^2]
    """

    # Master and individual toggles
    enabled: bool = True
    drag_enabled: bool = True
    lift_enabled: bool = True
    magnus_enabled: bool = True

    # Tunable coefficients
    drag_coefficient: float = float(GOLF_BALL_DRAG_COEFFICIENT)
    lift_coefficient: float = float(GOLF_BALL_LIFT_COEFFICIENT)
    magnus_coefficient: float = float(MAGNUS_COEFFICIENT)
    spin_decay_rate: float = float(SPIN_DECAY_RATE_S)

    # Advanced options
    reynolds_correction_enabled: bool = True

    # Ball properties
    ball_radius: float = float(GOLF_BALL_RADIUS_M)
    ball_area: float = float(GOLF_BALL_CROSS_SECTIONAL_AREA_M2)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.drag_coefficient < 0:
            raise ValueError("drag_coefficient must be non-negative")
        if self.lift_coefficient < 0:
            raise ValueError("lift_coefficient must be non-negative")
        if self.magnus_coefficient < 0:
            raise ValueError("magnus_coefficient must be non-negative")
        if self.spin_decay_rate < 0:
            raise ValueError("spin_decay_rate must be non-negative")

    def is_drag_active(self) -> bool:
        """Check if drag force is active."""
        return self.enabled and self.drag_enabled

    def is_lift_active(self) -> bool:
        """Check if lift force is active."""
        return self.enabled and self.lift_enabled

    def is_magnus_active(self) -> bool:
        """Check if Magnus force is active."""
        return self.enabled and self.magnus_enabled

    def with_changes(self, **kwargs: Any) -> AerodynamicsConfig:
        """Create a modified copy of this configuration (Reversible pattern).

        Args:
            **kwargs: Fields to modify

        Returns:
            New AerodynamicsConfig with specified changes
        """
        return replace(self, **kwargs)


@dataclass(frozen=True)
class WindConfig:
    """Configuration for wind model.

    Attributes:
        base_velocity: Constant wind velocity vector [m/s]
        gusts_enabled: Enable random gusts
        gust_intensity: Gust strength as fraction of base speed (0-1)
        gust_frequency: Average gust frequency [Hz]
        gust_duration_mean: Average gust duration [s]
        turbulence_intensity: Small-scale turbulence intensity
        altitude_gradient: Enable wind speed increase with altitude
        gradient_factor: Wind speed increase per 10m altitude
    """

    base_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    gusts_enabled: bool = False
    gust_intensity: float = 0.3
    gust_frequency: float = 0.1  # Hz
    gust_duration_mean: float = 2.0  # seconds
    turbulence_intensity: float = 0.0
    altitude_gradient: bool = False
    gradient_factor: float = 0.05  # 5% per 10m

    @property
    def speed(self) -> float:
        """Get base wind speed magnitude."""
        return float(np.linalg.norm(self.base_velocity))

    @property
    def direction(self) -> np.ndarray:
        """Get normalized wind direction."""
        speed = self.speed
        if speed < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return self.base_velocity / speed


@dataclass(frozen=True)
class RandomizationConfig:
    """Configuration for environment randomization.

    Attributes:
        enabled: Master switch for randomization
        air_density_variance: Relative variance in air density
        temperature_variance: Absolute variance in temperature [C]
        wind_variance: Relative variance in wind speed
        wind_direction_variance: Variance in wind direction [rad]
    """

    enabled: bool = False
    air_density_variance: float = 0.0
    temperature_variance: float = 0.0
    wind_variance: float = 0.0
    wind_direction_variance: float = 0.0


# =============================================================================
# Force Models (Orthogonal - each calculates one force type)
# =============================================================================


class DragModel:
    """Model for aerodynamic drag force.

    Drag opposes motion and scales with v^2:
        F_drag = -0.5 * rho * Cd * A * |v| * v

    The drag coefficient can optionally be corrected for Reynolds number,
    which accounts for the transition from laminar to turbulent flow
    around the golf ball's dimpled surface.
    """

    def __init__(
        self,
        base_coefficient: float = float(GOLF_BALL_DRAG_COEFFICIENT),
        ball_area: float = float(GOLF_BALL_CROSS_SECTIONAL_AREA_M2),
        ball_radius: float = float(GOLF_BALL_RADIUS_M),
        reynolds_correction: bool = True,
    ) -> None:
        """Initialize drag model.

        Args:
            base_coefficient: Base drag coefficient
            ball_area: Cross-sectional area [m^2]
            ball_radius: Ball radius [m]
            reynolds_correction: Apply Reynolds number correction
        """
        self.base_coefficient = base_coefficient
        self.ball_area = ball_area
        self.ball_radius = ball_radius
        self.reynolds_correction = reynolds_correction

    def calculate(
        self,
        velocity: np.ndarray,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> np.ndarray:
        """Calculate drag force.

        Args:
            velocity: Ball velocity relative to air [m/s]
            air_density: Air density [kg/m^3]

        Returns:
            Drag force vector [N]
        """
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-10:
            return np.zeros(3)

        cd = self.get_effective_coefficient(velocity, air_density)
        force_magnitude = 0.5 * air_density * cd * self.ball_area * speed**2

        # Drag opposes velocity
        return -force_magnitude * velocity / speed

    def get_effective_coefficient(
        self,
        velocity: np.ndarray,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> float:
        """Get drag coefficient, optionally corrected for Reynolds number.

        The Reynolds correction modifies the base coefficient based on flow regime:
        - Laminar flow (low Re): higher effective Cd
        - Turbulent flow (high Re): base coefficient applies
        - Transition: smooth interpolation

        Args:
            velocity: Ball velocity [m/s]
            air_density: Air density [kg/m^3]

        Returns:
            Effective drag coefficient
        """
        if not self.reynolds_correction:
            return self.base_coefficient

        speed = float(np.linalg.norm(velocity))
        if speed < 1e-10:
            return self.base_coefficient

        # Reynolds number
        viscosity = float(AIR_VISCOSITY_KG_M_S)
        diameter = 2 * self.ball_radius
        re = air_density * speed * diameter / viscosity

        # Golf ball Cd variation with Re (empirical)
        # The base_coefficient is used as the turbulent value, with
        # higher values at lower Reynolds numbers (laminar flow)
        laminar_cd = 0.5  # Laminar flow coefficient
        turbulent_cd = self.base_coefficient  # User-specified turbulent coefficient

        if re < 8e4:
            return laminar_cd  # Laminar flow
        elif re < 2e5:
            # Transition region - interpolate between laminar and turbulent
            fraction = (re - 8e4) / (2e5 - 8e4)
            return laminar_cd - fraction * (laminar_cd - turbulent_cd)
        else:
            return turbulent_cd  # Fully turbulent


class LiftModel:
    """Model for spin-induced lift force.

    Lift from backspin acts perpendicular to velocity in the
    plane defined by the spin axis and velocity vector.

        F_lift = 0.5 * rho * Cl * A * v^2 * lift_direction

    where lift_direction = normalize(spin_axis x velocity)
    """

    def __init__(
        self,
        base_coefficient: float = float(GOLF_BALL_LIFT_COEFFICIENT),
        ball_area: float = float(GOLF_BALL_CROSS_SECTIONAL_AREA_M2),
        ball_radius: float = float(GOLF_BALL_RADIUS_M),
        max_coefficient: float = 0.4,
    ) -> None:
        """Initialize lift model.

        Args:
            base_coefficient: Base lift coefficient
            ball_area: Cross-sectional area [m^2]
            ball_radius: Ball radius [m]
            max_coefficient: Maximum lift coefficient (saturation)
        """
        self.base_coefficient = base_coefficient
        self.ball_area = ball_area
        self.ball_radius = ball_radius
        self.max_coefficient = max_coefficient

    def calculate(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> np.ndarray:
        """Calculate lift force from spin.

        Args:
            velocity: Ball velocity [m/s]
            spin: Angular velocity [rad/s]
            air_density: Air density [kg/m^3]

        Returns:
            Lift force vector [N]
        """
        speed = float(np.linalg.norm(velocity))
        spin_magnitude = float(np.linalg.norm(spin))

        if speed < 1e-10 or spin_magnitude < 1e-10:
            return np.zeros(3)

        # Lift direction: perpendicular to velocity, in spin plane
        spin_axis = spin / spin_magnitude
        lift_dir = np.cross(spin_axis, velocity)
        lift_norm = float(np.linalg.norm(lift_dir))

        if lift_norm < 1e-10:
            return np.zeros(3)

        lift_dir = lift_dir / lift_norm

        # Effective lift coefficient based on spin ratio
        spin_ratio = self.ball_radius * spin_magnitude / speed
        cl = self._compute_lift_coefficient(spin_ratio)

        # Lift magnitude
        force_magnitude = 0.5 * air_density * cl * self.ball_area * speed**2

        return force_magnitude * lift_dir

    def _compute_lift_coefficient(self, spin_ratio: float) -> float:
        """Compute lift coefficient based on spin ratio.

        Args:
            spin_ratio: Dimensionless spin parameter (omega*R/v)

        Returns:
            Lift coefficient
        """
        # Empirical relationship: Cl saturates at high spin
        cl = self.max_coefficient * (1 - math.exp(-spin_ratio / 0.1))
        return min(cl, self.max_coefficient)


class MagnusModel:
    """Model for Magnus force from spin.

    The Magnus effect creates a force perpendicular to both velocity
    and spin axis, causing hook/slice for sidespin.

        F_magnus = 0.5 * rho * Cm * A * v^2 * (spin x velocity) / |spin x velocity|
    """

    def __init__(
        self,
        coefficient: float = float(MAGNUS_COEFFICIENT),
        ball_area: float = float(GOLF_BALL_CROSS_SECTIONAL_AREA_M2),
        ball_radius: float = float(GOLF_BALL_RADIUS_M),
    ) -> None:
        """Initialize Magnus model.

        Args:
            coefficient: Magnus coefficient
            ball_area: Cross-sectional area [m^2]
            ball_radius: Ball radius [m]
        """
        self.coefficient = coefficient
        self.ball_area = ball_area
        self.ball_radius = ball_radius

    def calculate(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> np.ndarray:
        """Calculate Magnus force.

        Args:
            velocity: Ball velocity [m/s]
            spin: Angular velocity [rad/s]
            air_density: Air density [kg/m^3]

        Returns:
            Magnus force vector [N]
        """
        speed = float(np.linalg.norm(velocity))
        spin_magnitude = float(np.linalg.norm(spin))

        if speed < 1e-10 or spin_magnitude < 1e-10:
            return np.zeros(3)

        # Magnus direction: spin x velocity
        magnus_dir = np.cross(spin, velocity)
        magnus_norm = float(np.linalg.norm(magnus_dir))

        if magnus_norm < 1e-10:
            return np.zeros(3)

        magnus_dir = magnus_dir / magnus_norm

        # Effective coefficient based on spin parameter
        spin_param = self.ball_radius * spin_magnitude / speed
        cm = self._compute_magnus_coefficient(spin_param)

        # Force magnitude
        force_magnitude = 0.5 * air_density * cm * self.ball_area * speed**2

        return force_magnitude * magnus_dir

    def _compute_magnus_coefficient(self, spin_param: float) -> float:
        """Compute Magnus coefficient based on spin parameter.

        Args:
            spin_param: Dimensionless spin parameter (omega*R/v)

        Returns:
            Magnus coefficient
        """
        # Approximately linear for small spin_param, saturates for large
        return self.coefficient * min(spin_param / 0.2, 1.0)


# =============================================================================
# Wind Models
# =============================================================================


class WindGust:
    """Single wind gust event with smooth envelope.

    A gust ramps up, holds at peak, then ramps down using
    a sinusoidal envelope for smooth transitions.
    """

    def __init__(
        self,
        start_time: float,
        duration: float,
        peak_velocity: np.ndarray,
    ) -> None:
        """Initialize gust event.

        Args:
            start_time: When gust begins [s]
            duration: Total gust duration [s]
            peak_velocity: Maximum gust velocity [m/s]
        """
        self.start_time = start_time
        self.duration = duration
        self.peak_velocity = peak_velocity

    @property
    def end_time(self) -> float:
        """Get gust end time."""
        return self.start_time + self.duration

    def get_velocity_at(self, t: float) -> np.ndarray:
        """Get gust velocity at time t.

        Uses a sinusoidal envelope for smooth transitions.

        Args:
            t: Time [s]

        Returns:
            Gust velocity at time t [m/s]
        """
        if t < self.start_time or t > self.end_time:
            return np.zeros(3)

        # Normalized time within gust (0 to 1)
        tau = (t - self.start_time) / self.duration

        # Sinusoidal envelope: sin^2 for smooth ramp up/down
        envelope = math.sin(math.pi * tau) ** 2

        return self.peak_velocity * envelope


class TurbulenceModel:
    """Small-scale atmospheric turbulence model.

    Uses Perlin-like noise for smooth, continuous turbulence.
    """

    def __init__(
        self,
        intensity: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize turbulence model.

        Args:
            intensity: Turbulence intensity scale [m/s]
            seed: Random seed for reproducibility
        """
        self.intensity = intensity
        self._rng = np.random.default_rng(seed)
        # Pre-generate noise coefficients for smooth interpolation
        self._coeffs = self._rng.standard_normal((3, 10))
        self._phases = self._rng.uniform(0, 2 * np.pi, (3, 10))
        self._freqs = self._rng.uniform(0.1, 2.0, 10)

    def get_perturbation(
        self,
        t: float,
        position: np.ndarray,
    ) -> np.ndarray:
        """Get turbulence perturbation at given time and position.

        Args:
            t: Time [s]
            position: Position [m]

        Returns:
            Turbulence velocity perturbation [m/s]
        """
        if self.intensity < 1e-10:
            return np.zeros(3)

        # Sum of sinusoids at different frequencies (poor man's Perlin noise)
        perturbation = np.zeros(3)
        for i in range(3):
            for j, freq in enumerate(self._freqs):
                perturbation[i] += self._coeffs[i, j] * math.sin(
                    freq * t + self._phases[i, j]
                )

        # Normalize and scale
        perturbation = perturbation / len(self._freqs) * self.intensity

        return perturbation


class WindModel:
    """Sophisticated wind model with gusts and turbulence.

    Features:
    - Constant base wind
    - Random gusts with configurable intensity and frequency
    - Small-scale turbulence
    - Altitude-dependent wind gradient (wind shear)
    - Reproducible with seed
    """

    def __init__(
        self,
        config: WindConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize wind model.

        Args:
            config: Wind configuration
            seed: Random seed for reproducibility
        """
        self.config = config or WindConfig()
        self._rng = np.random.default_rng(seed)
        self._gusts: list[WindGust] = []
        self._turbulence = TurbulenceModel(
            intensity=self.config.turbulence_intensity,
            seed=seed,
        )
        self._last_gust_time = -float("inf")
        self._last_check_time = -float("inf")
        self._dt_accumulator = 0.0  # Accumulate time for probabilistic spawning

    def get_wind_at(
        self,
        t: float,
        position: np.ndarray,
    ) -> np.ndarray:
        """Get wind velocity at given time and position.

        Args:
            t: Time [s]
            position: Position [m]

        Returns:
            Wind velocity [m/s]
        """
        # Start with base wind
        wind = self.config.base_velocity.copy()

        # Apply altitude gradient
        if self.config.altitude_gradient:
            altitude = max(0.0, position[2])
            gradient_multiplier = 1.0 + self.config.gradient_factor * (altitude / 10.0)
            wind = wind * gradient_multiplier

        # Add gusts
        if self.config.gusts_enabled:
            wind = wind + self._get_gust_contribution(t)

        # Add turbulence
        wind = wind + self._turbulence.get_perturbation(t, position)

        return wind

    def _get_gust_contribution(self, t: float) -> np.ndarray:
        """Get contribution from active gusts and possibly spawn new ones.

        Args:
            t: Time [s]

        Returns:
            Total gust velocity [m/s]
        """
        # Maybe spawn new gust
        self._maybe_spawn_gust(t)

        # Sum contributions from active gusts
        total = np.zeros(3)
        for gust in self._gusts:
            total = total + gust.get_velocity_at(t)

        # Clean up expired gusts
        self._gusts = [g for g in self._gusts if g.end_time > t]

        return total

    def _maybe_spawn_gust(self, t: float) -> None:
        """Possibly spawn a new gust event.

        Uses a Poisson process: expected gusts = frequency * time_elapsed.
        Accumulated time is used to properly handle irregular sampling.

        Args:
            t: Current time [s]
        """
        # Compute time since last check
        if self._last_check_time < 0:
            self._last_check_time = t
            dt = 0.1  # Initial time step assumption
        else:
            dt = t - self._last_check_time
            self._last_check_time = t

        if dt <= 0:
            return

        # Accumulate time for probabilistic spawning
        self._dt_accumulator += dt

        # Check spawning probability based on accumulated time
        # Expected gusts in accumulated time = frequency * accumulated_time
        spawn_probability = self.config.gust_frequency * self._dt_accumulator

        if self._rng.random() < spawn_probability:
            # Reset accumulator after spawn attempt
            self._dt_accumulator = 0.0

            # Generate random gust
            duration = self._rng.exponential(self.config.gust_duration_mean)
            duration = max(0.5, min(duration, 10.0))  # Clamp

            # Random direction perturbation
            base_speed = self.config.speed
            gust_speed = (
                base_speed * self.config.gust_intensity * self._rng.uniform(0.5, 1.5)
            )

            # Gust direction: mostly aligned with base wind, some random deviation
            base_dir = self.config.direction
            random_perturb = self._rng.standard_normal(3) * 0.3
            gust_dir = base_dir + random_perturb
            gust_dir = gust_dir / (np.linalg.norm(gust_dir) + 1e-10)

            gust = WindGust(
                start_time=t,
                duration=duration,
                peak_velocity=gust_dir * gust_speed,
            )
            self._gusts.append(gust)
            self._last_gust_time = t
        elif spawn_probability > 1.0:
            # If probability exceeds 1, reset to prevent unbounded growth
            self._dt_accumulator = 0.0


# =============================================================================
# Environment Randomization
# =============================================================================


@dataclass
class EnvironmentSnapshot:
    """Snapshot of randomized environment conditions.

    Provides consistent random values for a single simulation run.
    """

    air_density: float
    temperature: float
    wind_config: WindConfig | None = None


class EnvironmentRandomizer:
    """Randomize environment conditions for stochastic simulation.

    Provides reproducible randomization of:
    - Air density
    - Temperature
    - Wind speed and direction
    """

    def __init__(
        self,
        config: RandomizationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize randomizer.

        Args:
            config: Randomization configuration
            seed: Random seed for reproducibility
        """
        self.config = config or RandomizationConfig()
        self._rng = np.random.default_rng(seed)

    def randomize_air_density(self, base_density: float) -> float:
        """Randomize air density.

        Args:
            base_density: Base air density [kg/m^3]

        Returns:
            Randomized air density [kg/m^3]
        """
        if not self.config.enabled or self.config.air_density_variance <= 0:
            return base_density

        # Gaussian perturbation
        std = base_density * self.config.air_density_variance
        return float(self._rng.normal(base_density, std))

    def randomize_temperature(self, base_temperature: float) -> float:
        """Randomize temperature.

        Args:
            base_temperature: Base temperature [C]

        Returns:
            Randomized temperature [C]
        """
        if not self.config.enabled or self.config.temperature_variance <= 0:
            return base_temperature

        return float(
            self._rng.normal(base_temperature, self.config.temperature_variance)
        )

    def randomize_wind_config(self, base_config: WindConfig) -> WindConfig:
        """Randomize wind configuration.

        Args:
            base_config: Base wind configuration

        Returns:
            Randomized wind configuration
        """
        if not self.config.enabled:
            return base_config

        # Randomize speed
        base_speed = base_config.speed
        if self.config.wind_variance > 0 and base_speed > 0:
            speed_std = base_speed * self.config.wind_variance
            new_speed = float(self._rng.normal(base_speed, speed_std))
            new_speed = max(0.0, new_speed)  # Non-negative
        else:
            new_speed = base_speed

        # Randomize direction
        base_dir = base_config.direction
        if self.config.wind_direction_variance > 0:
            # Rotate by random angle
            angle = float(self._rng.normal(0, self.config.wind_direction_variance))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            # Rotate in xy-plane
            new_dir = np.array(
                [
                    cos_a * base_dir[0] - sin_a * base_dir[1],
                    sin_a * base_dir[0] + cos_a * base_dir[1],
                    base_dir[2],
                ]
            )
        else:
            new_dir = base_dir

        new_velocity = new_dir * new_speed

        # Create new config with randomized velocity
        return WindConfig(
            base_velocity=new_velocity,
            gusts_enabled=base_config.gusts_enabled,
            gust_intensity=base_config.gust_intensity,
            gust_frequency=base_config.gust_frequency,
            gust_duration_mean=base_config.gust_duration_mean,
            turbulence_intensity=base_config.turbulence_intensity,
            altitude_gradient=base_config.altitude_gradient,
            gradient_factor=base_config.gradient_factor,
        )

    def create_snapshot(
        self,
        base_air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
        base_temperature: float = 15.0,
        base_wind_config: WindConfig | None = None,
    ) -> EnvironmentSnapshot:
        """Create a consistent randomized environment snapshot.

        Args:
            base_air_density: Base air density [kg/m^3]
            base_temperature: Base temperature [C]
            base_wind_config: Base wind configuration

        Returns:
            Randomized environment snapshot
        """
        return EnvironmentSnapshot(
            air_density=self.randomize_air_density(base_air_density),
            temperature=self.randomize_temperature(base_temperature),
            wind_config=(
                self.randomize_wind_config(base_wind_config)
                if base_wind_config
                else None
            ),
        )


# =============================================================================
# Unified Aerodynamics Engine
# =============================================================================


class AerodynamicsEngine:
    """Unified aerodynamics calculation engine.

    Combines all aerodynamic force models with optional wind and
    environment randomization. All effects can be toggled on/off.

    Example:
        >>> config = AerodynamicsConfig(drag_enabled=True, lift_enabled=True)
        >>> engine = AerodynamicsEngine(config)
        >>> forces = engine.compute_forces(velocity, spin)
        >>> print(forces['total'])
    """

    def __init__(
        self,
        config: AerodynamicsConfig | None = None,
        wind_model: WindModel | None = None,
        randomization: EnvironmentRandomizer | None = None,
        air_density: float = float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    ) -> None:
        """Initialize aerodynamics engine.

        Args:
            config: Aerodynamics configuration
            wind_model: Wind model for variable wind
            randomization: Environment randomizer
            air_density: Base air density [kg/m^3]
        """
        self.config = config or AerodynamicsConfig()
        self.wind_model = wind_model
        self.randomization = randomization
        self._base_air_density = air_density
        self._current_air_density = air_density

        # Initialize force models
        self._drag = DragModel(
            base_coefficient=self.config.drag_coefficient,
            ball_area=self.config.ball_area,
            ball_radius=self.config.ball_radius,
            reynolds_correction=self.config.reynolds_correction_enabled,
        )
        self._lift = LiftModel(
            base_coefficient=self.config.lift_coefficient,
            ball_area=self.config.ball_area,
            ball_radius=self.config.ball_radius,
        )
        self._magnus = MagnusModel(
            coefficient=self.config.magnus_coefficient,
            ball_area=self.config.ball_area,
            ball_radius=self.config.ball_radius,
        )

    def compute_forces(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
        t: float = 0.0,
        position: np.ndarray | None = None,
        resample: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute all aerodynamic forces.

        Args:
            velocity: Ball velocity [m/s]
            spin: Angular velocity [rad/s]
            t: Current time [s] (for wind variation)
            position: Current position [m] (for wind gradient)
            resample: Resample random environment

        Returns:
            Dictionary with 'drag', 'lift', 'magnus', and 'total' forces [N]
        """
        if position is None:
            position = np.zeros(3)

        # Resample air density if randomization enabled
        if resample and self.randomization:
            self._current_air_density = self.randomization.randomize_air_density(
                self._base_air_density
            )

        # Get wind velocity
        wind = np.zeros(3)
        if self.wind_model:
            wind = self.wind_model.get_wind_at(t, position)

        # Relative velocity (ball velocity minus wind)
        rel_velocity = velocity - wind

        # Initialize forces
        drag = np.zeros(3)
        lift = np.zeros(3)
        magnus = np.zeros(3)

        # Compute active forces
        if self.config.is_drag_active():
            drag = self._drag.calculate(rel_velocity, self._current_air_density)

        if self.config.is_lift_active():
            lift = self._lift.calculate(rel_velocity, spin, self._current_air_density)

        if self.config.is_magnus_active():
            magnus = self._magnus.calculate(
                rel_velocity, spin, self._current_air_density
            )

        total = drag + lift + magnus

        return {
            "drag": drag,
            "lift": lift,
            "magnus": magnus,
            "total": total,
        }

    def compute_acceleration(
        self,
        velocity: np.ndarray,
        spin: np.ndarray,
        mass: float,
        t: float = 0.0,
        position: np.ndarray | None = None,
        resample: bool = False,
    ) -> np.ndarray:
        """Compute acceleration from aerodynamic forces.

        Args:
            velocity: Ball velocity [m/s]
            spin: Angular velocity [rad/s]
            mass: Ball mass [kg]
            t: Current time [s]
            position: Current position [m]
            resample: Resample random environment

        Returns:
            Acceleration vector [m/s^2]
        """
        forces = self.compute_forces(velocity, spin, t, position, resample)
        return forces["total"] / mass

    def compute_spin_decay(
        self,
        spin: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute spin decay over time step.

        Spin decays exponentially due to air resistance.

        Args:
            spin: Current angular velocity [rad/s]
            dt: Time step [s]

        Returns:
            Updated spin after decay [rad/s]
        """
        decay_factor = math.exp(-self.config.spin_decay_rate * dt)
        return spin * decay_factor
