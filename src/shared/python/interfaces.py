"""Unified Abstract Interface for Physics Engines.

This module defines the Protocol that all physics engines (MuJoCo, Drake, Pinocchio, etc.)
must adhere to. This ensures that the GUI and Analytics layers can operate
agnostic of the underlying solver.

Design by Contract:
    This interface defines contracts that all implementations must satisfy:

    State Machine:
        UNINITIALIZED -> [load_from_path/load_from_string] -> INITIALIZED
        INITIALIZED -> [reset] -> INITIALIZED (t=0)
        INITIALIZED -> [step] -> INITIALIZED (t+=dt)

    Global Invariants (all implementations must maintain):
        - After initialization: model is loaded and queryable
        - Time is always non-negative
        - State arrays (q, v) have consistent dimensions
        - Mass matrix is always symmetric positive definite
        - Superposition: a_full = a_drift + a_control (Section F)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.checkpoint import Checkpointable


@runtime_checkable
class PhysicsEngine(Checkpointable, Protocol):
    """Protocol defining the required interface for a Golf Modeling Suite physics engine.

    All implementations must be stateless wrappers around a Model/Data pair (or equivalent),
    or manage their own internal state consistently.

    Design by Contract:
        This protocol defines the contract between the simulation framework and
        physics engine implementations. Each method documents its:
        - Preconditions: What must be true before calling
        - Postconditions: What will be true after successful return
        - Invariants: What is preserved by the operation
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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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

    # -------- Dynamics Interface --------

    @abstractmethod
    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q).

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - Returns symmetric positive definite matrix
            - M.shape == (n_v, n_v)
            - All values are finite
            - M == M.T (symmetric)
            - All eigenvalues > 0 (positive definite)

        Returns:
            M: (n_v, n_v) mass matrix.

        Raises:
            StateError: If engine is not initialized
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v) + g(q).

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - b.shape == (n_v,)
            - All values are finite

        Returns:
            b: (n_v,) vector containing Coriolis, Centrifugal, and Gravity terms.

        Raises:
            StateError: If engine is not initialized
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q).

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - g.shape == (n_v,)
            - All values are finite

        Returns:
            g: (n_v,) gravity vector.

        Raises:
            StateError: If engine is not initialized
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics tau = ID(q, v, a).

        Preconditions:
            - Engine must be in INITIALIZED state
            - qacc.shape == (n_v,)
            - qacc must contain finite values

        Postconditions:
            - tau.shape == (n_v,)
            - tau = M(q) @ qacc + C(q,v) @ v + g(q)
            - All values are finite

        Args:
            qacc: Desired acceleration vector (n_v,).

        Returns:
            tau: Required generalized forces (n_v,).

        Raises:
            StateError: If engine is not initialized
            ValueError: If qacc has wrong dimensions
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body.

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - Returns None if body_name not found
            - Otherwise returns dict with 'linear' (3, n_v) and 'angular' (3, n_v)
            - All values are finite

        Args:
            body_name: Name of the body frame.

        Returns:
            Dictionary with keys 'linear', 'angular', 'spatial', or None if body not found.

        Raises:
            StateError: If engine is not initialized
        """
        raise NotImplementedError()

    def compute_contact_forces(self) -> np.ndarray:
        """Compute total contact forces (GRF).

        Preconditions:
            - Engine must be in INITIALIZED state

        Postconditions:
            - f.shape == (3,) or (6,)
            - All values are finite

        Returns:
            f: (3,) vector representing total ground reaction force,
               or (6,) wrench (force + torque) if supported.
               Default implementation returns zero vector.
        """
        return np.zeros(3)

    # -------- Section F: Drift-Control Decomposition (Non-Negotiable) --------

    @abstractmethod
    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (drift) acceleration with zero control inputs.

        Section F Requirement: Drift component = passive dynamics (Coriolis, centrifugal, gravity, constraints)
        with all applied torques/muscle activations set to zero.

        Mathematically: q̈_drift = M(q)⁻¹ · (C(q,v)v + g(q))

        This is the answer to: "What would happen if all motors/muscles turned off right now?"

        Preconditions:
            - Engine must be in INITIALIZED state
            - State (q, v) must be set

        Postconditions:
            - a_drift.shape == (n_v,)
            - All values are finite
            - CRITICAL CONTRACT: a_drift + a_control = a_full (superposition)

        Returns:
            q_ddot_drift: Drift acceleration vector (n_v,) [rad/s² or m/s²]

        Raises:
            StateError: If engine is not initialized

        See Also:
            - compute_control_acceleration: Control-attributed component
            - Section F: Superposition requirement (drift + control = full)
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute control-attributed acceleration from applied torques/forces only.

        Section F Requirement: Control component = acceleration due solely to actuator torques,
        excluding passive dynamics.

        Mathematically: q̈_control = M(q)⁻¹ · τ

        Preconditions:
            - Engine must be in INITIALIZED state
            - tau.shape == (n_v,)
            - tau must contain finite values

        Postconditions:
            - a_control.shape == (n_v,)
            - All values are finite
            - CRITICAL CONTRACT: a_drift + a_control = a_full (superposition)

        Args:
            tau: Applied generalized forces/torques (n_v,) [N·m or N]

        Returns:
            q_ddot_control: Control acceleration vector (n_v,) [rad/s² or m/s²]

        Raises:
            StateError: If engine is not initialized
            ValueError: If tau has wrong dimensions

        Note:
            For muscle-driven models, tau represents muscle-generated joint torques.
        """
        raise NotImplementedError()

    # -------- Section G: Counterfactual Experiments (Mandatory) --------

    @abstractmethod
    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compute acceleration with applied torques set to zero, preserving current state.
        This isolates drift (gravity + Coriolis + constraints) from control effects.

        **Purpose**: Answer "What would happen if all actuators turned off RIGHT NOW?"

        **Physics**: With τ=0, acceleration is purely passive:
            q̈_ZTCF = M(q)⁻¹ · (C(q,v)·v + g(q) + J^T·λ)

        **Causal Interpretation**:
            Δa_control = a_full - a_ZTCF
            This is the acceleration *attributed to* actuator torques.

        Preconditions:
            - Engine must be in INITIALIZED state
            - q.shape == (n_q,), v.shape == (n_v,)
            - q and v must contain finite values

        Postconditions:
            - a_ztcf.shape == (n_v,)
            - All values are finite
            - CRITICAL CONTRACT: At current state, ZTCF == drift acceleration

        **Example Use Case** (Golf Swing):
            At impact, compute ZTCF to determine how much clubhead acceleration
            is due to passive dynamics (arm falling under gravity + centrifugal)
            vs. active muscle torques.

        Args:
            q: Joint positions (n_v,) [rad or m]
            v: Joint velocities (n_v,) [rad/s or m/s]

        Returns:
            q̈_ZTCF: Acceleration under zero applied torque (n_v,) [rad/s² or m/s²]

        Raises:
            StateError: If engine is not initialized
            ValueError: If array dimensions don't match model

        Note:
            State (q, v) is preserved; only applied control is zeroed.
            Constraints remain active (J^T·λ term preserved).

        See Also:
            - compute_zvcf: Zero-velocity counterfactual
            - Section G1: ZTCF definition in design guidelines
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compute acceleration with joint velocities set to zero, preserving configuration.
        This isolates configuration-dependent effects (gravity, constraints)
        from velocity-dependent effects (Coriolis, centrifugal).

        **Purpose**: Answer "What acceleration would occur if motion FROZE instantaneously?"

        **Physics**: With v=0, acceleration has no velocity-dependent terms:
            q̈_ZVCF = M(q)⁻¹ · (g(q) + τ + J^T·λ)

        **Causal Interpretation**:
            Δa_velocity = a_full - a_ZVCF
            This is the acceleration *attributed to* Coriolis/centrifugal effects.

        Preconditions:
            - Engine must be in INITIALIZED state
            - q.shape == (n_q,)
            - q must contain finite values

        Postconditions:
            - a_zvcf.shape == (n_v,)
            - All values are finite
            - No velocity-dependent terms in result

        **Example Use Case** (Golf Swing):
            During downswing, compute ZVCF to separate gravitational pull
            from centrifugal whip effect. At fast velocities, Coriolis dominates.

        Args:
            q: Joint positions (n_v,) [rad or m]

        Returns:
            q̈_ZVCF: Acceleration with v=0 (n_v,) [rad/s² or m/s²]

        Raises:
            StateError: If engine is not initialized
            ValueError: If array dimensions don't match model

        Note:
            Only velocity is zeroed; configuration (q) and control (τ) preserved.
            Centrifugal barrier analysis uses ZVCF to find configurations where
            q̈(q,0,τ) prevents motion even with applied torque.

        See Also:
            - compute_ztcf: Zero-torque counterfactual
            - Section G2: ZVCF definition in design guidelines
        """
        raise NotImplementedError()

    # ---------------------------------------------------------------------------
    # Section B5: Flexible Beam Shaft (Optional Interface)
    # ---------------------------------------------------------------------------

    def set_shaft_properties(
        self,
        length: float,
        EI_profile: np.ndarray,
        mass_profile: np.ndarray,
        damping_ratio: float = 0.02,
    ) -> bool:
        """Configure flexible shaft properties (Guideline B5).

        This is an OPTIONAL method. Engines that don't support flexible shafts
        should return False.

        Args:
            length: Total shaft length [m]
            EI_profile: Bending stiffness at each station [N·m²] (n_stations,)
            mass_profile: Mass per unit length at each station [kg/m] (n_stations,)
            damping_ratio: Modal damping ratio [unitless], default 0.02

        Returns:
            True if shaft properties were successfully configured, False otherwise.

        Note:
            The shaft model is engine-dependent:
            - MuJoCo: Composite body chain with torsional joints
            - Drake: Multibody with compliant elements
            - Pinocchio: Modal representation

        Example:
            >>> from shared.python.flexible_shaft import create_standard_shaft, compute_EI_profile
            >>> props = create_standard_shaft(ShaftMaterial.GRAPHITE)
            >>> EI = compute_EI_profile(props)
            >>> mass = compute_mass_profile(props)
            >>> success = engine.set_shaft_properties(props.length, EI, mass)
        """
        # Default implementation returns False (not supported)
        return False

    def get_shaft_state(self) -> dict[str, np.ndarray] | None:
        """Get current shaft deformation state.

        Returns:
            Dictionary with:
            - 'deflection': Transverse deflection at each station [m] (n_stations,)
            - 'rotation': Local rotation at each station [rad] (n_stations,)
            - 'velocity': Transverse velocity at each station [m/s] (n_stations,)
            - 'modal_amplitudes': Modal amplitude for each mode (n_modes,)

            Returns None if shaft flexibility is not configured.
        """
        return None


@runtime_checkable
class RecorderInterface(Protocol):
    """Protocol for recording and retrieving simulation data.

    Allows different backends (MuJoCo, Drake, Pinocchio) to be visualized
    using the same widgets.
    """

    # The engine associated with this recorder (optional, but useful for joint names)
    engine: Any

    @abstractmethod
    def get_time_series(
        self, field_name: str
    ) -> tuple[np.ndarray, np.ndarray | list[Any]]:
        """Get time series data for a specific field.

        Args:
            field_name: The metric key (e.g. 'joint_positions', 'ztcf_accel').

        Returns:
            Tuple of (times, values).
            times: (N,) array of time timestamps.
            values: (N, D) array of data values.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    @abstractmethod
    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Configure which advanced metrics to record/compute.

        Args:
            config: Dictionary of configuration flags (e.g. {'ztcf': True}).
        """
        raise NotImplementedError()
