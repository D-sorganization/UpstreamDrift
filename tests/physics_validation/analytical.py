"""Analytical solutions for physics validation baselines."""

import numpy as np


class AnalyticalPendulum:
    """Exact solution for a simple pendulum."""

    def __init__(self, length: float = 1.0, mass: float = 1.0, g: float = 9.81, inertia: float | None = None):
        self.L = length
        self.m = mass
        self.g = g
        # Default to point mass inertia if not specified
        self.I = inertia if inertia is not None else (mass * length**2)

    def potential_energy(self, theta: float) -> float:
        """Calculate potential energy relative to bottom position.

        PE = m * g * h
        h = L * (1 - cos(theta))
        """
        h = self.L * (1.0 - np.cos(theta))
        return self.m * self.g * h

    def kinetic_energy(self, omega: float) -> float:
        """Calculate kinetic energy.

        KE = 0.5 * I * omega^2
        """
        return 0.5 * self.I * omega**2

    def total_energy(self, theta: float, omega: float) -> float:
        """Calculate total mechanical energy."""
        return self.potential_energy(theta) + self.kinetic_energy(omega)


class AnalyticalBallistic:
    """Exact solution for a ballistic trajectory (no drag)."""

    def __init__(self, mass: float = 1.0, g: float = 9.81):
        self.m = mass
        self.g = g

    def total_energy(self, height: float, velocity: float) -> float:
        """Calculate total energy.

        E = PE + KE = mgh + 0.5mv^2
        """
        pe = self.m * self.g * height
        ke = 0.5 * self.m * velocity**2
        return pe + ke
