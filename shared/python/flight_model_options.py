"""Flight model configuration and feature flags.

Provides optional features (wind, spin decay) that can be enabled
without changing the tuned coefficients of the original models.
"""

import math
from dataclasses import dataclass


@dataclass
class FlightModelOptions:
    """Optional features for flight models.

    These features are OFF by default to preserve original model
    calibrations. Enable with caution as they may change trajectory
    predictions from the original source implementations.
    """

    # Wind effects
    enable_wind: bool = False  # Apply wind drag if True

    # Spin decay
    enable_spin_decay: bool = False  # Apply exponential spin decay if True
    spin_decay_rate: float = 0.05  # Decay rate [1/s] (τ ≈ 20s half-life)

    # Altitude effects
    enable_altitude_correction: bool = False
    altitude_m: float = 0.0  # Altitude above sea level [m]


def compute_spin_decay(
    omega_initial: float, time: float, decay_rate: float
) -> float:
    """Compute spin rate with exponential decay.

    Spin decays as: ω(t) = ω₀ * exp(-λ*t)

    Args:
        omega_initial: Initial angular velocity [rad/s]
        time: Time since launch [s]
        decay_rate: Decay constant λ [1/s]

    Returns:
        Decayed angular velocity [rad/s]
    """
    return omega_initial * math.exp(-decay_rate * time)


def compute_air_density_at_altitude(
    sea_level_density: float, altitude_m: float
) -> float:
    """Compute air density at altitude using barometric formula.

    Uses simplified isothermal atmosphere model.

    Args:
        sea_level_density: Air density at sea level [kg/m³]
        altitude_m: Altitude above sea level [m]

    Returns:
        Air density at altitude [kg/m³]
    """
    # Scale height for isothermal atmosphere ≈ 8500m
    scale_height = 8500.0
    return sea_level_density * math.exp(-altitude_m / scale_height)


# Default options instance (all features off)
DEFAULT_OPTIONS = FlightModelOptions()
