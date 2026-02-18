"""Pinocchio Physics Engine wrapper implementation.

Wraps pinocchio to provide a compliant PhysicsEngine interface.

Inherits from BasePhysicsEngine to eliminate DRY violations for checkpoint
save/restore, model loading boilerplate, model name tracking, and
initialization patterns.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.shared.python.core.contracts import (
    check_finite,
    invariant,
    postcondition,
    precondition,
)
from src.shared.python.engine_core.base_physics_engine import (
    BasePhysicsEngine,
)
from src.shared.python.engine_core.engine_availability import (
    PINOCCHIO_AVAILABLE,
)
from src.shared.python.logging_pkg.logging_config import get_logger

# Pinocchio imports - only import if available
if PINOCCHIO_AVAILABLE:
    import pinocchio as pin

from src.shared.python.core import constants

logger = get_logger(__name__)

DEFAULT_TIME_STEP = float(constants.DEFAULT_TIME_STEP)


@invariant(
    lambda self: self.model is None or self.data is not None,
    "If model is loaded, data must also be initialized",
)
@invariant(
    lambda self: self.time >= 0.0,
    "Simulation time must be non-negative",
)
class PinocchioPhysicsEngine(BasePhysicsEngine):
    """Encapsulates Pinocchio model, data, and simulation control.

    Implements the shared PhysicsEngine protocol via BasePhysicsEngine.

    Inherits common functionality from BasePhysicsEngine:
    - Model loading with path validation and error handling
    - Checkpoint save/restore (protocol-compatible path)
    - Model name tracking (uses pinocchio model.name)
    - String representation
    """

    def __init__(self) -> None:
        """Initialize the Pinocchio physics engine."""
        super().__init__()

        # State arrays (pinocchio manages own state, not EngineState)
        self.q: np.ndarray = np.array([])
        self.v: np.ndarray = np.array([])
        self.a: np.ndarray = np.array([])
        self.tau: np.ndarray = np.array([])
        self.time: float = 0.0

    @property
    def is_initialized(self) -> bool:
        """Check if the engine has a loaded model and data."""
        return self.model is not None and self.data is not None

    @property
    def engine_type(self) -> str:
        """Get engine type identifier."""
        return "pinocchio"

    @property
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        if self.model:
            return cast(str, self.model.name)
        return self.model_name_str

    def _load_from_path_impl(self, path: str) -> None:
        """Pinocchio-specific model loading from URDF file path.

        Args:
            path: Validated path to URDF model file.
        """
        if not path.endswith(".urdf"):
            logger.warning("Pinocchio loader expects URDF, got: %s", path)

        self.model = pin.buildModelFromUrdf(path)
        self.data = self.model.createData()
        self.model_name_str = self.model.name

        # Initialize state
        self.q = pin.neutral(self.model)
        self.v = np.zeros(self.model.nv)
        self.a = np.zeros(self.model.nv)
        self.tau = np.zeros(self.model.nv)
        self.time = 0.0

    def _load_from_string_impl(self, content: str, extension: str | None) -> None:
        """Pinocchio-specific model loading from XML string.

        Args:
            content: Model definition string (URDF/XML).
            extension: File extension hint.
        """
        if extension != "urdf":
            logger.warning("Pinocchio load_from_string mostly supports URDF.")

        self.model = pin.buildModelFromXML(content)
        self.data = self.model.createData()
        self.model_name_str = "StringLoadedModel"

        self.q = pin.neutral(self.model)
        self.v = np.zeros(self.model.nv)
        self.a = np.zeros(self.model.nv)
        self.tau = np.zeros(self.model.nv)
        self.time = 0.0

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
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

    @precondition(
        lambda self, dt=None: self.is_initialized,
        "Engine must be initialized",
    )
    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one time step."""
        if self.model is None or self.data is None:
            return

        time_step = dt if dt is not None else DEFAULT_TIME_STEP

        # Explicit Forward Dynamics: a = ABA(q, v, tau)
        self.a = pin.aba(self.model, self.data, self.q, self.v, self.tau)

        # Semi-implicit Euler integration
        self.v += self.a * time_step
        self.q = pin.integrate(self.model, self.q, self.v * time_step)

        self.time += time_step

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
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

        names = list(self.model.names)
        if "universe" in names:
            names.remove("universe")
        return names

    def get_full_state(self) -> dict[str, Any]:
        """Get complete state in a single batched call.

        Returns:
            Dictionary with 'q', 'v', 't', and 'M' (mass matrix).
        """
        if self.model is None or self.data is None:
            return {
                "q": np.array([]),
                "v": np.array([]),
                "t": 0.0,
                "M": None,
            }

        q = self.q.copy()
        v = self.v.copy()
        t = self.time

        pin.crba(self.model, self.data, self.q)

        # Symmetrize
        M = self.data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T

        return {"q": q, "v": v, "t": t, "M": M}

    # -------- Dynamics Interface --------

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Mass matrix must contain finite values")
    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        pin.crba(self.model, self.data, self.q)

        # Symmetrize
        M = self.data.M.copy()
        M = np.triu(M) + np.triu(M, 1).T
        return cast(np.ndarray, M)

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Bias forces must contain finite values")
    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v) + g(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        a_zero = np.zeros(self.model.nv)
        return cast(
            np.ndarray,
            pin.rnea(self.model, self.data, self.q, self.v, a_zero),
        )

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Gravity forces must contain finite values")
    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        return cast(
            np.ndarray,
            pin.computeGeneralizedGravity(self.model, self.data, self.q),
        )

    @precondition(
        lambda self, qacc: self.is_initialized,
        "Engine must be initialized",
    )
    @postcondition(
        check_finite,
        "Inverse dynamics torques must contain finite values",
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
            Returns placeholder zero vector. Pinocchio's standard ABA
            does not compute contact forces without a constraint solver.

        Returns:
            f: (3,) zero vector (placeholder).
        """
        if self.data is None:
            return np.zeros(3)

        logger.warning(
            "PinocchioPhysicsEngine.compute_contact_forces currently "
            "returns a placeholder zero GRF vector. Standard ABA "
            "dynamics in Pinocchio do not compute contact forces "
            "without a constraint solver. "
            "This is a known limitation of this wrapper."
        )

        return np.zeros(3)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body."""
        if self.model is None or self.data is None:
            return None

        if not self.model.existFrame(body_name):
            logger.warning(f"Body/Frame '{body_name}' not found in Pinocchio model.")
            return None

        frame_id = self.model.getFrameId(body_name)

        J = pin.getFrameJacobian(
            self.model,
            self.data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        # J is (6, nv): Linear, Angular ordering
        jac_linear = J[:3, :]
        jac_angular = J[3:, :]

        # Standardize on [Angular; Linear] for "spatial" key
        J_aligned = np.vstack([jac_angular, jac_linear])

        return {
            "linear": cast(np.ndarray, jac_linear),
            "angular": cast(np.ndarray, jac_angular),
            "spatial": cast(np.ndarray, J_aligned),
        }

    # -------- Section F: Drift-Control Decomposition --------

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Drift acceleration must contain finite values")
    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (drift) acceleration with zero control.

        Uses Pinocchio's ABA with zero torque.

        Returns:
            q_ddot_drift: Drift acceleration vector (nv,)
        """
        if self.model is None or self.data is None:
            return np.array([])

        tau_zero = np.zeros(self.model.nv)
        a_drift = pin.aba(self.model, self.data, self.q, self.v, tau_zero)

        return cast(np.ndarray, a_drift)

    @precondition(
        lambda self, tau: self.is_initialized,
        "Engine must be initialized",
    )
    @postcondition(check_finite, "Control acceleration must contain finite values")
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute control-attributed acceleration: M(q)^-1 * tau.

        Args:
            tau: Applied generalized forces (nv,)

        Returns:
            q_ddot_control: Control acceleration vector (nv,)
        """
        if self.model is None or self.data is None:
            return np.array([])

        if len(tau) != self.model.nv:
            return np.array([])

        M = self.compute_mass_matrix()
        if M.size == 0:
            return np.array([])

        a_control = np.linalg.solve(M, tau)

        return a_control

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compute acceleration with applied torques set to zero.

        Args:
            q: Joint positions (n_q,) [rad or m]
            v: Joint velocities (n_v,) [rad/s or m/s]

        Returns:
            q_ddot_ZTCF: Acceleration under zero torque (n_v,)
        """
        if self.model is None or self.data is None:
            return np.array([])

        if len(q) != self.model.nq or len(v) != self.model.nv:
            return np.array([])

        tau_zero = np.zeros(self.model.nv)
        a_ztcf = pin.aba(self.model, self.data, q, v, tau_zero)

        return cast(np.ndarray, a_ztcf)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compute acceleration with joint velocities set to zero.

        Args:
            q: Joint positions (n_q,) [rad or m]

        Returns:
            q_ddot_ZVCF: Acceleration with v=0 (n_v,)
        """
        if self.model is None or self.data is None:
            return np.array([])

        if len(q) != self.model.nq:
            return np.array([])

        v_zero = np.zeros(self.model.nv)

        # Use current control (preserved for ZVCF)
        tau = self.tau.copy()

        a_zvcf = pin.aba(self.model, self.data, q, v_zero, tau)

        return cast(np.ndarray, a_zvcf)
