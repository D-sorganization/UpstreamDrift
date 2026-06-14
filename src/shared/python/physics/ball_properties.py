"""Ball physical properties and aerodynamic coefficient calculations.

This submodule contains the BallProperties dataclass and related constants
extracted from ball_flight_physics.py as part of P1 sprint decomposition
(issue #2486).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.physics_constants import (
    GOLF_BALL_DIAMETER_M,
    GOLF_BALL_MASS_KG,
    SPIN_DECAY_RATE_S,
)

MIN_SPEED_THRESHOLD: float = 0.1
MAX_LIFT_COEFFICIENT: float = 0.26
PENNER_LIFT_SCALE: float = 0.70
PENNER_LIFT_EXPONENT: float = 0.645
NUMERICAL_EPSILON: float = 1e-10


def calculate_spin_lift_coefficient(s: float) -> float:
    """Compute a bounded Penner-style lift coefficient from spin ratio."""
    if s is None:
        raise ValueError("spin parameter must be provided")
    if s <= 0.0:
        return 0.0
    return min(MAX_LIFT_COEFFICIENT, PENNER_LIFT_SCALE * s**PENNER_LIFT_EXPONENT)


@dataclass(frozen=True)
class BallProperties:
    """Physical properties of a golf ball (DRY-optimized)."""

    mass: float = float(GOLF_BALL_MASS_KG)
    diameter: float = float(GOLF_BALL_DIAMETER_M)
    cd0: float = 0.21
    cd1: float = 0.25
    cd2: float = 0.02
    cl0: float = 0.00
    cl1: float = 0.38
    cl2: float = 0.08
    spin_decay_rate: float = float(SPIN_DECAY_RATE_S)

    @property
    def radius(self) -> float:
        """Return the ball radius in meters."""
        return self.diameter / 2

    @property
    def cross_sectional_area(self) -> float:
        """Return the cross-sectional area of the ball."""
        return float(np.pi * (self.diameter / 2) ** 2)

    def calculate_cd(self, s: float) -> float:
        """Compute the drag coefficient from the spin parameter."""
        return self.cd0 + s * (self.cd1 + s * self.cd2)

    def calculate_cl(self, s: float) -> float:
        """Compute the shared spin-lift coefficient from the spin parameter."""
        return calculate_spin_lift_coefficient(float(s))
