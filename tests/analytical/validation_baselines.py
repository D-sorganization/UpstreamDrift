"""Analytical helper baselines for physics validation tests."""

from __future__ import annotations

import numpy as np

from src.shared.python.core.constants import GRAVITY_M_S2


class AnalyticalPendulum:
    """Closed-form energy terms for a simple pendulum."""

    def __init__(
        self,
        length: float = 1.0,
        mass: float = 1.0,
        g: float = GRAVITY_M_S2,
        inertia: float | None = None,
    ) -> None:
        assert length is not None, "length must be provided"
        assert mass is not None, "mass must be provided"
        assert g is not None, "g must be provided"
        assert length > 0.0, "length must be positive"
        assert mass > 0.0, "mass must be positive"
        self.L = length
        self.m = mass
        self.g = g
        self.I = inertia if inertia is not None else (mass * length**2)

    def potential_energy(self, theta: float) -> float:
        """Calculate potential energy relative to the bottom position."""
        assert theta is not None, "theta must be provided"
        assert isinstance(theta, int | float), "theta must be a number"
        h = self.L * (1.0 - np.cos(theta))
        return float(self.m * self.g * h)

    def kinetic_energy(self, omega: float) -> float:
        """Calculate kinetic energy."""
        return 0.5 * self.I * omega**2

    def total_energy(self, theta: float, omega: float) -> float:
        """Calculate total mechanical energy."""
        assert theta is not None, "theta must be provided"
        assert omega is not None, "omega must be provided"
        assert isinstance(theta, int | float), "theta must be a number"
        assert isinstance(omega, int | float), "omega must be a number"
        return self.potential_energy(theta) + self.kinetic_energy(omega)


class AnalyticalBallistic:
    """Closed-form energy terms for a ballistic trajectory without drag."""

    def __init__(self, mass: float = 1.0, g: float = GRAVITY_M_S2) -> None:
        assert mass is not None, "mass must be provided"
        self.m = mass
        self.g = g

    def total_energy(self, height: float, velocity: float) -> float:
        """Calculate total mechanical energy."""
        assert height is not None, "height must be provided"
        assert velocity is not None, "velocity must be provided"
        assert isinstance(height, int | float), "height must be numeric"
        assert isinstance(velocity, int | float), "velocity must be numeric"
        pe = self.m * self.g * height
        ke = 0.5 * self.m * velocity**2
        return pe + ke
