# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Abstract physics backend interface.

Provides a clean abstraction layer so the optimizer can work with
different physics engines.  The current implementation uses analytical
Lagrangian dynamics; future backends may wrap MuJoCo, Drake, or
Pinocchio for full 3-D multi-body simulations.

Design notes (Law of Demeter):
    Consumers of a backend should only call methods defined in
    ``PhysicsBackend``.  They should never reach into the backend
    to access internal solver state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray


class PhysicsBackend(ABC):
    """Common interface every dynamics engine must implement.

    All methods operate on generalised-coordinate vectors *q* whose
    length equals the number of degrees of freedom (currently 3 for
    the sagittal-plane chain: shin, thigh, torso).
    """

    @abstractmethod
    def forward_kinematics(self, q: NDArray) -> dict[str, NDArray]:
        """Return named joint positions {name: (x, y)} for pose *q*."""

    @abstractmethod
    def bar_position(self, q: NDArray, exercise_type: str) -> NDArray:
        """Return (x, y) barbell position for the given pose."""

    @abstractmethod
    def com_position(
        self,
        q: NDArray,
        exercise_type: str = "squat",
        bar_mass: float = 0.0,
    ) -> NDArray:
        """Return (x, y) whole-body centre of mass."""

    @abstractmethod
    def inverse_dynamics(self, q: NDArray, qd: NDArray, qdd: NDArray) -> NDArray:
        """Compute joint torques from motion: tau = M*qdd + C + G."""

    @abstractmethod
    def inverse_dynamics_batch(self, q: NDArray, qd: NDArray, qdd: NDArray) -> NDArray:
        """Vectorised batch torques: q, qd, qdd are (N, n_dof)."""

    @abstractmethod
    def com_x_batch(self, q: NDArray, exercise_type: str, bar_mass: float) -> NDArray:
        """Vectorised batch COM x-coordinate: q is (N, n_dof)."""

    @abstractmethod
    def mass_matrix(self, q: NDArray) -> NDArray:
        """Return the n x n mass/inertia matrix M(q)."""

    @property
    @abstractmethod
    def segment_lengths(self) -> NDArray:
        """Lengths of the active kinematic chain segments."""

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
