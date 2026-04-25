from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.engine_core.checkpoint import Checkpointable

if TYPE_CHECKING:
    from src.shared.python.engine_core.capabilities import EngineCapabilities

__all__ = [
    "SimulationInterface",
]


@runtime_checkable
class SimulationInterface(Checkpointable, Protocol):
    """Sub-protocol covering model loading, stepping, state query, and metadata."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the currently loaded model.

        Preconditions:
            - None (can be called at any time)

        Postconditions:
            - Returns empty string if no model loaded
            - Returns model identifier if model is loaded
        """
        ...

    @abstractmethod
    def load_from_path(self, path: str) -> None:
        """Load a model from a file path.

        Preconditions:
            - path must be a valid file path
            - path must point to an existing file
            - file must be in a supported format (.xml, .urdf, .sdf, .osim)

        Postconditions:
            - Engine is in INITIALIZED state
            - model_name returns valid identifier
            - get_state() returns valid arrays
            - Invariants are established

        Args:
            path: Absolute path to the model file (.xml, .urdf, .sdf, .osim).

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If file format is not supported
            StateError: If engine cannot be initialized
        """
        ...

    @abstractmethod
    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load a model from a string content.

        Preconditions:
            - content must be non-empty
            - content must be valid model definition

        Postconditions:
            - Engine is in INITIALIZED state
            - get_state() returns valid arrays

        Args:
            content: The model definition string.
            extension: Optional hint for parsing (e.g., 'xml', 'urdf').

        Raises:
            ValueError: If content is empty or invalid
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation to its initial state (time=0, q=q0, v=0).

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - get_time() == 0.0
            - State is at initial configuration

        Raises:
            StateError: If engine is not initialized
        """
        ...

    @abstractmethod
    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one time step.

        Preconditions:
            - Engine must be in INITIALIZED state
            - dt > 0 if provided

        Postconditions:
            - get_time() increased by dt
            - State updated according to dynamics
            - All derived quantities recomputed

        Args:
            dt: Optional time step to advance. If None, uses the model's default timestep.

        Raises:
            StateError: If engine is not initialized
            ValueError: If dt <= 0
        """
        ...

    @abstractmethod
    def forward(self) -> None:
        """Compute forward kinematics and dynamics without advancing time.

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - All derived quantities updated (accelerations, forces)
            - Time unchanged
            - State (q, v) unchanged

        Using current positions and velocities, updates all derived quantities
        (accelerations, forces, derived kinematics).

        Raises:
            StateError: If engine is not initialized
        """
        ...

    @abstractmethod
    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities).

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - Returns tuple of (q, v) numpy arrays
            - q.shape == (n_q,), v.shape == (n_v,)
            - Arrays contain finite values (no NaN/Inf)

        Returns:
            Tuple of (q, v) as numpy arrays.
            q: Generalized coordinates (n_q,).
            v: Generalized velocities (n_v,).

        Raises:
            StateError: If engine is not initialized
        """
        ...

    @abstractmethod
    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state.

        Preconditions:
            - Engine must be in INITIALIZED state
            - q.shape == (n_q,), v.shape == (n_v,)
            - Arrays must contain finite values

        Postconditions:
            - get_state() returns (q, v)
            - Derived quantities updated via forward()

        Args:
            q: Generalized coordinates.
            v: Generalized velocities.

        Raises:
            StateError: If engine is not initialized
            ValueError: If array dimensions don't match model
        """
        ...

    @abstractmethod
    def set_control(self, u: np.ndarray) -> None:
        """Apply control inputs (torques/forces).

        Preconditions:
            - Engine must be in INITIALIZED state
            - u.shape == (n_u,)
            - Array must contain finite values

        Postconditions:
            - Control stored for next step/forward call

        Args:
            u: Control vector (n_u,).

        Raises:
            StateError: If engine is not initialized
            ValueError: If array dimension doesn't match model
        """
        ...

    @abstractmethod
    def get_time(self) -> float:
        """Get the current simulation time.

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - Returns time >= 0.0

        Returns:
            Current simulation time in seconds.

        Raises:
            StateError: If engine is not initialized
        """
        ...

    def get_full_state(self) -> dict[str, Any]:
        """Get complete state in a single batched call (performance optimization).

        This method reduces the overhead of multiple separate engine queries by
        returning all commonly-needed state information in one call.

        Returns:
            Dictionary containing:
            - 'q': Generalized coordinates (n_q,)
            - 'v': Generalized velocities (n_v,)
            - 't': Current simulation time
            - 'M': Mass matrix (n_v, n_v) - optional, may be None if expensive

        Note:
            Default implementation calls individual methods. Engines should
            override this for better performance if they can batch these queries.
        """
        q, v = self.get_state()
        return {
            "q": q,
            "v": v,
            "t": self.get_time(),
            "M": None,  # Default: don't compute expensive mass matrix
        }

    def get_joint_names(self) -> list[str]:
        """Get list of joint names.

        Returns:
            List of strings corresponding to the joint names in order.
            Default implementation returns generic names.
        """
        return []

    def get_capabilities(self) -> EngineCapabilities:
        """Report which optional capabilities this engine supports.

        Returns an EngineCapabilities dataclass describing the support level
        for each optional feature (video export, dataset export, force
        visualization, model positioning, measurements).

        Engines should override this method to accurately report their
        capabilities. The default returns NONE for all optional features.

        Returns:
            EngineCapabilities with support levels for each feature.
        """
        from src.shared.python.engine_core.capabilities import (
            CapabilityLevel,
            EngineCapabilities,
        )

        return EngineCapabilities(
            engine_name="unknown",
            mass_matrix=CapabilityLevel.NONE,
            jacobian=CapabilityLevel.NONE,
            contact_forces=CapabilityLevel.NONE,
            inverse_dynamics=CapabilityLevel.NONE,
            drift_acceleration=CapabilityLevel.NONE,
        )
