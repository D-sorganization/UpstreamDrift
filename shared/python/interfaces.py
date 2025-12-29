"""Unified Abstract Interface for Physics Engines.

This module defines the Protocol that all physics engines (MuJoCo, Drake, Pinocchio, etc.)
must adhere to. This ensures that the GUI and Analytics layers can operate
agnostic of the underlying solver.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PhysicsEngine(Protocol):
    """Protocol defining the required interface for a Golf Modeling Suite physics engine.

    All implementations must be stateless wrappers around a Model/Data pair (or equivalent),
    or manage their own internal state consistently.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        ...

    @abstractmethod
    def load_from_path(self, path: str) -> None:
        """Load a model from a file path.

        Args:
            path: Absolute path to the model file (.xml, .urdf, .sdf, .osim).
        """
        ...

    @abstractmethod
    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load a model from a string content.

        Args:
            content: The model definition string.
            extension: Optional hint for parsing (e.g., 'xml', 'urdf').
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation to its initial state (time=0, q=q0, v=0)."""
        ...

    @abstractmethod
    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one time step.

        Args:
            dt: Optional time step to advance. If None, uses the model's default timestep.
        """
        ...

    @abstractmethod
    def forward(self) -> None:
        """Compute forward kinematics and dynamics without advancing time.

        Using current positions and velocities, updates all derived quantities
        (accelerations, forces, derived kinematics).
        """
        ...

    @abstractmethod
    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities).

        Returns:
            Tuple of (q, v) as numpy arrays.
            q: Generalized coordinates (n_q,).
            v: Generalized velocities (n_v,).
        """
        ...

    @abstractmethod
    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state.

        Args:
            q: Generalized coordinates.
            v: Generalized velocities.
        """
        ...

    @abstractmethod
    def set_control(self, u: np.ndarray) -> None:
        """Apply control inputs (torques/forces).

        Args:
            u: Control vector (n_u,).
        """
        ...

    @abstractmethod
    def get_time(self) -> float:
        """Get the current simulation time."""
        ...

    # -------- Dynamics Interface --------

    @abstractmethod
    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q).

        Returns:
            M: (n_v, n_v) mass matrix.
        """
        ...

    @abstractmethod
    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v) + g(q).

        Returns:
            b: (n_v,) vector containing Coriolis, Centrifugal, and Gravity terms.
        """
        ...

    @abstractmethod
    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q).

        Returns:
            g: (n_v,) gravity vector.
        """
        ...

    @abstractmethod
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics tau = ID(q, v, a).

        Args:
            qacc: Desired acceleration vector (n_v,).

        Returns:
            tau: Required generalized forces (n_v,).
        """
        ...

    @abstractmethod
    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body.

        Args:
            body_name: Name of the body frame.

        Returns:
            Dictionary with keys 'linear', 'angular', 'spatial', or None if body not found.
        """
        ...
