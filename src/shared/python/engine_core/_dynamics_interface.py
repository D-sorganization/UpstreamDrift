from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "DynamicsInterface",
]


@runtime_checkable
class DynamicsInterface(Protocol):
    """Sub-protocol covering dynamics computation, counterfactuals, and shaft properties."""

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
