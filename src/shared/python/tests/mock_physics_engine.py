"""Mock physics engine for testing."""

from __future__ import annotations

from typing import cast

import numpy as np

from src.shared.python.interfaces import PhysicsEngine


class MockPhysicsEngine:
    """Mock physics engine for testing purposes."""

    def __init__(self, n_dof: int = 2):
        """Initialize mock engine.

        Args:
            n_dof: Number of degrees of freedom
        """
        self.n_dof = n_dof
        self.time = 0.0
        self.q = np.zeros(n_dof)
        self.v = np.zeros(n_dof)
        self.mass_matrix = np.eye(n_dof)
        self.gravity_forces = np.zeros(n_dof)

    def get_time(self) -> float:
        """Get current simulation time."""
        return self.time

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current state (q, v)."""
        return self.q.copy(), self.v.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set state (q, v)."""
        self.q = q.copy()
        self.v = v.copy()

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute mass matrix M(q)."""
        return self.mass_matrix.copy()

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q)."""
        return self.gravity_forces.copy()

    def set_mass_matrix(self, M: np.ndarray) -> None:
        """Set mass matrix for testing."""
        self.mass_matrix = M.copy()

    def set_gravity_forces(self, g: np.ndarray) -> None:
        """Set gravity forces for testing."""
        self.gravity_forces = g.copy()

    def advance_time(self, dt: float) -> None:
        """Advance simulation time."""
        self.time += dt


def as_physics_engine(mock: MockPhysicsEngine) -> PhysicsEngine:
    """Cast mock engine to PhysicsEngine interface for type checking."""
    return cast(PhysicsEngine, mock)
