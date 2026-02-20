"""Composable sub-protocols for the PhysicsEngine interface.

This module decomposes the monolithic PhysicsEngine Protocol (22 abstract methods)
into focused, composable sub-protocols following the Interface Segregation Principle.

Each sub-protocol is @runtime_checkable and can be used independently for
partial interface checks. The full PhysicsEngine Protocol in interfaces.py
remains as the composition of all sub-protocols for backward compatibility.

Sub-protocols:
    Loadable: Model loading and identification (3 methods)
    Steppable: Time stepping (3 methods)
    Queryable: State inspection (4 methods)
    DynamicsComputable: Physics computation (7 methods)
    CounterfactualComputable: What-if analysis (2 methods)
    Recordable: Data collection (3 methods)

Design by Contract:
    Each sub-protocol documents its own preconditions and postconditions.
    Implementations may satisfy any subset of sub-protocols.
    The full PhysicsEngine Protocol requires all sub-protocols to be satisfied.

Example:
    >>> from src.shared.python.engine_core.sub_protocols import Steppable, Queryable
    >>> def run_simulation(engine: Steppable & Queryable, steps: int) -> None:
    ...     for _ in range(steps):
    ...         engine.step()
    ...     q, v = engine.get_state()
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Sub-protocol 1: Loadable - Model loading and identification
# ---------------------------------------------------------------------------


@runtime_checkable
class Loadable(Protocol):
    """Protocol for model loading and identification.

    Implementations must support loading models from file paths or string content,
    and provide a model_name property for identification.

    State Machine:
        UNINITIALIZED -> [load_from_path/load_from_string] -> INITIALIZED
    """

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

        Args:
            path: Absolute path to the model file.

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If file format is not supported
        """
        ...

    @abstractmethod
    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load a model from string content.

        Preconditions:
            - content must be non-empty
            - content must be valid model definition

        Postconditions:
            - Engine is in INITIALIZED state

        Args:
            content: The model definition string.
            extension: Optional hint for parsing (e.g., 'xml', 'urdf').

        Raises:
            ValueError: If content is empty or invalid
        """
        ...


# ---------------------------------------------------------------------------
# Sub-protocol 2: Steppable - Time stepping
# ---------------------------------------------------------------------------


@runtime_checkable
class Steppable(Protocol):
    """Protocol for time stepping operations.

    Implementations must support reset, step, and forward kinematics/dynamics
    computation without advancing time.

    State Machine:
        INITIALIZED -> [reset] -> INITIALIZED (t=0)
        INITIALIZED -> [step] -> INITIALIZED (t+=dt)
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation to its initial state (time=0, q=q0, v=0).

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - Time is reset to 0.0
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
            - Time increased by dt
            - State updated according to dynamics

        Args:
            dt: Optional time step. If None, uses model default.

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
            - All derived quantities updated
            - Time unchanged
            - State (q, v) unchanged

        Raises:
            StateError: If engine is not initialized
        """
        ...


# ---------------------------------------------------------------------------
# Sub-protocol 3: Queryable - State inspection
# ---------------------------------------------------------------------------


@runtime_checkable
class Queryable(Protocol):
    """Protocol for state inspection and modification.

    Implementations must support getting and setting the generalized state
    (positions q, velocities v), setting control inputs, and querying time.

    Invariants:
        - State arrays (q, v) have consistent dimensions
        - Time is always non-negative
    """

    @abstractmethod
    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities).

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - Returns tuple of (q, v) numpy arrays
            - q.shape == (n_q,), v.shape == (n_v,)
            - Arrays contain finite values

        Returns:
            Tuple of (q, v) as numpy arrays.

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


# ---------------------------------------------------------------------------
# Sub-protocol 4: DynamicsComputable - Physics computation
# ---------------------------------------------------------------------------


@runtime_checkable
class DynamicsComputable(Protocol):
    """Protocol for dynamics computation.

    Implementations must support computing mass matrices, bias forces,
    gravity forces, inverse dynamics, Jacobians, and drift-control decomposition.

    Invariants:
        - Mass matrix is always symmetric positive definite
        - Superposition: a_full = a_drift + a_control (Section F)
    """

    @abstractmethod
    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q).

        Postconditions:
            - Returns symmetric positive definite matrix
            - M.shape == (n_v, n_v)

        Returns:
            M: (n_v, n_v) mass matrix.
        """
        ...

    @abstractmethod
    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v) + g(q).

        Postconditions:
            - b.shape == (n_v,)

        Returns:
            b: (n_v,) bias force vector.
        """
        ...

    @abstractmethod
    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q).

        Postconditions:
            - g.shape == (n_v,)

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
            Dictionary with 'linear', 'angular' keys, or None if body not found.
        """
        ...

    @abstractmethod
    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (drift) acceleration with zero control inputs.

        Section F Requirement: q_ddot_drift = M(q)^-1 * (C(q,v)v + g(q))

        Returns:
            q_ddot_drift: Drift acceleration vector (n_v,).
        """
        ...

    @abstractmethod
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute control-attributed acceleration from applied torques/forces.

        Section F Requirement: q_ddot_control = M(q)^-1 * tau

        Args:
            tau: Applied generalized forces/torques (n_v,).

        Returns:
            q_ddot_control: Control acceleration vector (n_v,).
        """
        ...


# ---------------------------------------------------------------------------
# Sub-protocol 5: CounterfactualComputable - What-if analysis
# ---------------------------------------------------------------------------


@runtime_checkable
class CounterfactualComputable(Protocol):
    """Protocol for counterfactual experiments (Section G).

    Implementations must support zero-torque and zero-velocity counterfactual
    computations for causal analysis of swing dynamics.
    """

    @abstractmethod
    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compute acceleration with applied torques set to zero.
        Isolates drift (gravity + Coriolis + constraints) from control.

        Args:
            q: Joint positions (n_q,).
            v: Joint velocities (n_v,).

        Returns:
            q_ddot_ZTCF: Acceleration under zero applied torque (n_v,).
        """
        ...

    @abstractmethod
    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compute acceleration with joint velocities set to zero.
        Isolates configuration-dependent effects from velocity-dependent.

        Args:
            q: Joint positions (n_q,).

        Returns:
            q_ddot_ZVCF: Acceleration with v=0 (n_v,).
        """
        ...


# ---------------------------------------------------------------------------
# Sub-protocol 6: Recordable - Data collection
# ---------------------------------------------------------------------------


@runtime_checkable
class Recordable(Protocol):
    """Protocol for recording and retrieving simulation data.

    Implementations must support time series retrieval, induced acceleration
    analysis, and analysis configuration.
    """

    @abstractmethod
    def get_time_series(
        self, field_name: str
    ) -> tuple[np.ndarray, np.ndarray | list[Any]]:
        """Get time series data for a specific field.

        Args:
            field_name: The metric key (e.g. 'joint_positions', 'ztcf_accel').

        Returns:
            Tuple of (times, values).
        """
        ...

    @abstractmethod
    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get time series for induced acceleration from a specific source.

        Args:
            source_name: Name of the source (e.g. 'gravity') or actuator index.

        Returns:
            Tuple of (times, values).
        """
        ...

    @abstractmethod
    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Configure which advanced metrics to record/compute.

        Args:
            config: Dictionary of configuration flags (e.g. {'ztcf': True}).
        """
        ...
