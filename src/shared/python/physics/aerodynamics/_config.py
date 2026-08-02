"""Aerodynamics configuration dataclasses.

Contains immutable configuration classes for aerodynamic effects,
wind modelling, and environment randomization.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from src.shared.python.core.physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    GOLF_BALL_CROSS_SECTIONAL_AREA_M2,
    GOLF_BALL_DRAG_COEFFICIENT,
    GOLF_BALL_LIFT_COEFFICIENT,
    GOLF_BALL_RADIUS_M,
    MAGNUS_COEFFICIENT,
    SPIN_DECAY_RATE_S,
)

MIN_AIR_DENSITY_KG_M3 = 0.01


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
    lift_enabled: bool = False
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


__all__ = [
    "AerodynamicsConfig",
    "AIR_DENSITY_SEA_LEVEL_KG_M3",
    "RandomizationConfig",
    "WindConfig",
]
