"""Pinocchio Physics Engine wrapper implementation.

Wraps pinocchio to provide a compliant PhysicsEngine interface.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

# Pinocchio imports likely to fail if not installed
try:
    import pinocchio as pin
except ImportError:
    pass

from src.shared.python import constants
from src.shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)

DEFAULT_TIME_STEP = float(constants.DEFAULT_TIME_STEP)


class PinocchioPhysicsEngine(PhysicsEngine):
    """Encapsulates Pinocchio model, data, and simulation control.

    Implements the shared PhysicsEngine protocol.
    """

    def __init__(self) -> None:
        """Initialize the Pinocchio physics engine."""
        self.model: pin.Model | None = None
        self.data: pin.Data | None = None
        self.model_path: str = ""
        self.model_name_str: str = ""

        # State
        self.q: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.a: np.ndarray = np.array([])
        self.tau: np.ndarray = np.array([])
        self.time: float = 0.0

    @property
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        if self.model:
            return cast(str, self.model.name)
        return self.model_name_str

    def load_from_path(self, path: str) -> None:
        """Load model from file path (URDF)."""
        # Pinocchio typically loads URDFs
        if not path.endswith(".urdf"):
            LOGGER.warning("Pinocchio loader expects URDF, got: %s", path)

        try:
            self.model = pin.buildModelFromUrdf(path)
            self.data = self.model.createData()
            self.model_path = path
            self.model_name_str = self.model.name

            # Initialize state
            self.q = pin.neutral(self.model)
            self.v = np.zeros(self.model.nv)
            self.a = np.zeros(self.model.nv)
            self.tau = np.zeros(self.model.nv)
            self.time = 0.0

        except Exception as e:
            LOGGER.error("Failed to load Pinocchio model from path %s: %s", path, e)
            raise

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from string content."""
        if extension != "urdf":
            LOGGER.warning("Pinocchio load_from_string mostly supports URDF.")

        try:
            self.model = pin.buildModelFromXML(content)
            self.data = self.model.createData()
            self.model_name_str = "StringLoadedModel"

            self.q = pin.neutral(self.model)
            self.v = np.zeros(self.model.nv)
            self.a = np.zeros(self.model.nv)
            self.tau = np.zeros(self.model.nv)
            self.time = 0.0

        except Exception as e:
            LOGGER.error("Failed to load Pinocchio model from string: %s", e)
            raise

    def reset(self) -> None:
        """Reset the simulation to its initial state."""
        if self.model:
            self.q = pin.neutral(self.model)
            self.v = np.zeros(self.model.nv)
            self.a = np.zeros(self.model.nv)
            self.tau = np.zeros(self.model.nv)
            self.time = 0.0
            # Refresh data
            self.forward()

    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one time step."""
        if self.model is None or self.data is None:
            return

        # Use provided dt or default to standard
        time_step = dt if dt is not None else DEFAULT_TIME_STEP

        # Explicit Forward Dynamics
        # a = ABA(q, v, tau)
        self.a = pin.aba(self.model, self.data, self.q, self.v, self.tau)

        # Semi-implicit Euler integration
        # v_next = v + a * dt
        # q_next = integrate(q, v_next * dt)

        self.v += self.a * time_step
        self.q = pin.integrate(self.model, self.q, self.v * time_step)

        self.time += time_step

    def forward(self) -> None:
        """Compute forward kinematics/dynamics without advancing time."""
        if self.model is None or self.data is None:
            return

        pin.forwardKinematics(self.model, self.data, self.q, self.v, self.a)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities)."""
        return self.q.copy(), self.v.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state."""
        if self.model is None:
            return

        if len(q) == self.model.nq:
            self.q = q.copy()
        if len(v) == self.model.nv:
            self.v = v.copy()

    def set_control(self, u: np.ndarray) -> None:
        """Apply control inputs (torques/forces)."""
        if self.model is None:
            return
        if len(u) == self.model.nv:
            self.tau = u.copy()

    def get_time(self) -> float:
        """Get the current simulation time."""
        return self.time

    def get_joint_names(self) -> list[str]:
        """Get list of joint names."""
        if self.model is None:
            return []

        # Pinocchio model.names is a vector of strings
        # But it includes "universe" usually.
        # We want names corresponding to tangent vector v/tau?
        # Actually model.names corresponds to joints (nq).
        # Tangent vector corresponds to nv.

        # This is a simplification.
        names = list(self.model.names)
        if "universe" in names:
            names.remove("universe")
        return names

    def get_full_state(self) -> dict[str, Any]:
        """Get complete state in a single batched call (performance optimization).

        PERFORMANCE FIX: Returns all commonly-needed state in one call to avoid
        multiple separate engine queries.

        Returns:
            Dictionary with 'q', 'v', 't', and 'M' (mass matrix).
        """
        if self.model is None or self.data is None:
            return {"q": np.array([]), "v": np.array([]), "t": 0.0, "M": None}

        # Get state
        q = self.q.copy()
        v = self.v.copy()
        t = self.time

        # Compute mass matrix
        # CRBA computes the upper triangular part of the joint space inertia matrix
        pin.crba(self.model, self.data, self.q)

        # Symmetrize
        M = self.data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T

        return {"q": q, "v": v, "t": t, "M": M}

    # -------- Dynamics Interface --------

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        # CRBA computes the upper triangular part of the joint space inertia
        # matrix and stores it in data.M
        pin.crba(self.model, self.data, self.q)

        # Symmetrize
        M = self.data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T
        return cast(np.ndarray, M)

    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v) + g(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        # rnea(q, v, 0) -> M a + b
        # If a=0, result is b = C(q,v) + g(q)
        # Standard implementation
        a_zero = np.zeros(self.model.nv)
        return cast(
            np.ndarray,
            pin.rnea(self.model, self.data, self.q, self.v, a_zero),
        )

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        # computeGeneralizedGravity(model, data, q)
        return cast(
            np.ndarray, pin.computeGeneralizedGravity(self.model, self.data, self.q)
        )

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics tau = ID(q, v, a)."""
        if self.model is None or self.data is None:
            return np.array([])

        tau = pin.rnea(self.model, self.data, self.q, self.v, qacc)
        return cast(np.ndarray, tau)

    def compute_contact_forces(self) -> np.ndarray:
        """Compute total contact forces (ground reaction force, GRF).

        Notes:
            This Pinocchio wrapper currently returns a placeholder zero vector for the
            GRF. Pinocchio's standard forward dynamics (ABA) does not natively
            compute contact forces without additional constraint solver setup, which
            is not currently implemented in this lightweight wrapper.

            If you need accurate GRFs, you would need to implement a constraint
            dynamics solver or use a physics engine that supports contact natively
            in its standard step (like MuJoCo).

        Returns:
            f: (3,) vector representing total ground reaction force (currently
                always zeros as a placeholder).
        """
        if self.data is None:
            return np.zeros(3)

        # Pinocchio stores constraint forces in data.lambda_c if solver is used.
        # But for generic forward dynamics (ABA), contact forces are not computed
        # unless we use a contact solver (like constraint dynamics).

        # If simulation uses simple fwd dynamics without explicit contacts, return 0.
        # Pinocchio's standard forward dynamics doesn't handle contacts natively
        # without extra setup (e.g. Proximal or KKT).

        LOGGER.warning(
            "PinocchioPhysicsEngine.compute_contact_forces currently returns a "
            "placeholder zero GRF vector. Standard ABA dynamics in Pinocchio do "
            "not compute contact forces without a constraint solver. "
            "This is a known limitation of this wrapper."
        )

        return np.zeros(3)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body."""
        if self.model is None or self.data is None:
            return None

        # Simplified lookup: Check frame existence first
        if not self.model.existFrame(body_name):
            LOGGER.warning(f"Body/Frame '{body_name}' not found in Pinocchio model.")
            return None

        frame_id = self.model.getFrameId(body_name)

        # computeJointJacobians needs to be called first (done in forward)
        # getFrameJacobian
        J = pin.getFrameJacobian(
            self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # J is (6, nv) with Pinocchio Motion ordering: Linear, Angular.
        jac_linear = J[:3, :]
        jac_angular = J[3:, :]

        # Standardizing on [Angular; Linear] for "spatial" output key.
        J_aligned = np.vstack([jac_angular, jac_linear])

        return {
            "linear": cast(np.ndarray, jac_linear),
            "angular": cast(np.ndarray, jac_angular),
            "spatial": cast(np.ndarray, J_aligned),
        }

    # -------- Section F: Drift-Control Decomposition --------

    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (drift) acceleration with zero control inputs.

        Section F Implementation: Uses Pinocchio's ABA (Articulated Body Algorithm)
        with zero torque to compute passive dynamics due to gravity and
        Coriolis/centrifugal forces.

        Returns:
            q_ddot_drift: Drift acceleration vector (nv,) [rad/s² or m/s²]
        """
        if self.model is None or self.data is None:
            return np.array([])

        # Zero torque forward dynamics = drift only
        tau_zero = np.zeros(self.model.nv)
        a_drift = pin.aba(self.model, self.data, self.q, self.v, tau_zero)

        return cast(np.ndarray, a_drift)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute control-attributed acceleration from applied torques only.

        Section F Implementation: Computes M(q)^-1 * tau to isolate control component.

        Args:
            tau: Applied generalized forces (nv,) [N·m or N]

        Returns:
            q_ddot_control: Control acceleration vector (nv,) [rad/s² or m/s²]
        """
        if self.model is None or self.data is None:
            return np.array([])

        # Ensure tau has correct dimensions
        if len(tau) != self.model.nv:
            return np.array([])

        # Get mass matrix
        M = self.compute_mass_matrix()
        if M.size == 0:
            return np.array([])

        # Control component: M^-1 * tau
        a_control = np.linalg.solve(M, tau)

        return a_control

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compute acceleration with applied torques set to zero, preserving state.
        This isolates drift (gravity + Coriolis) from control effects.

        **Purpose**: Answer "What would happen if all actuators turned off?"

        **Physics**: With τ=0, acceleration is purely passive:
            q̈_ZTCF = ABA(q, v, 0) = M(q)⁻¹ · (-C(q,v) - g(q))

        Args:
            q: Joint positions (n_q,) [rad or m]
            v: Joint velocities (n_v,) [rad/s or m/s]

        Returns:
            q̈_ZTCF: Acceleration under zero applied torque (n_v,) [rad/s² or m/s²]
        """
        if self.model is None or self.data is None:
            return np.array([])

        # Validate dimensions
        if len(q) != self.model.nq or len(v) != self.model.nv:
            return np.array([])

        # Use Pinocchio's ABA with zero torque - this is exactly ZTCF
        tau_zero = np.zeros(self.model.nv)
        a_ztcf = pin.aba(self.model, self.data, q, v, tau_zero)

        return cast(np.ndarray, a_ztcf)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compute acceleration with joint velocities set to zero, preserving
        configuration. This isolates configuration-dependent effects (gravity)
        from velocity-dependent effects (Coriolis, centrifugal).

        **Purpose**: Answer "What acceleration would occur if motion FROZE?"

        **Physics**: With v=0, acceleration has no velocity-dependent terms:
            q̈_ZVCF = ABA(q, 0, τ) = M(q)⁻¹ · (-g(q) + τ)

        Args:
            q: Joint positions (n_q,) [rad or m]

        Returns:
            q̈_ZVCF: Acceleration with v=0 (n_v,) [rad/s² or m/s²]
        """
        if self.model is None or self.data is None:
            return np.array([])

        # Validate dimensions
        if len(q) != self.model.nq:
            return np.array([])

        # Use zero velocity for ZVCF
        v_zero = np.zeros(self.model.nv)

        # Use current control (preserved for ZVCF)
        tau = self.tau.copy()

        # Use ABA with zero velocity - this is ZVCF
        a_zvcf = pin.aba(self.model, self.data, q, v_zero, tau)

        return cast(np.ndarray, a_zvcf)
