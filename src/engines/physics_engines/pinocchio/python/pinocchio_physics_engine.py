"""Pinocchio Physics Engine wrapper implementation.

Wraps pinocchio to provide a compliant PhysicsEngine interface.

Inherits from BasePhysicsEngine to eliminate DRY violations for checkpoint
save/restore, model loading boilerplate, model name tracking, and
initialization patterns.
"""

from __future__ import annotations

from typing import Literal, cast

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
from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
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
PinocchioIntegrator = Literal["semi_implicit", "rk4"]


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
        self.integrator: PinocchioIntegrator = "rk4"

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

    def _require_vector(
        self,
        name: str,
        value: np.ndarray,
        expected_length: int,
    ) -> np.ndarray:
        """Validate that an input is a one-dimensional vector of expected length."""
        if value is None:
            raise ValueError(f"{name} must be provided")

        value_arr = np.asarray(value, dtype=np.float64)
        if value_arr.ndim != 1:
            raise ValueError(
                f"{name} dimension mismatch: expected length {expected_length}, "
                f"got shape {value_arr.shape}"
            )

        actual_length = int(value_arr.shape[0])
        if actual_length != expected_length:
            raise ValueError(
                f"{name} dimension mismatch: expected length {expected_length}, "
                f"got {actual_length}"
            )

        return value_arr

    def get_capabilities(self) -> EngineCapabilities:
        """Report Pinocchio capabilities, including lack of contact GRF support."""
        return EngineCapabilities(
            engine_name="Pinocchio",
            mass_matrix=CapabilityLevel.FULL,
            jacobian=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.NONE,
            inverse_dynamics=CapabilityLevel.FULL,
            drift_acceleration=CapabilityLevel.FULL,
            extra={"spatial_jacobian_order": "angular_linear"},
        )

    def _load_from_path_impl(self, path: str) -> None:
        """Pinocchio-specific model loading from URDF file path.

        Args:
            path: Validated path to URDF model file.
        """
        if not (path is not None):
            raise ValueError("path must be provided")
        if not (path is not None):
            raise ValueError("path must be provided")
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
        if not (content is not None):
            raise ValueError("content must be provided")
        if not (content is not None):
            raise ValueError("content must be provided")
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
        lambda self, dt=None, **kwargs: self.is_initialized,
        "Engine must be initialized",
    )
    def step(
        self,
        dt: float | None = None,
        *,
        integrator: PinocchioIntegrator | None = None,
    ) -> None:
        """Advance the simulation by one time step."""
        if self.model is None or self.data is None:
            return

        time_step = dt if dt is not None else DEFAULT_TIME_STEP
        if time_step <= 0.0:
            raise ValueError("dt must be positive")

        method = self.integrator if integrator is None else integrator
        if method == "rk4":
            self._step_rk4(time_step)
        elif method == "semi_implicit":
            self._step_semi_implicit(time_step)
        else:
            raise ValueError(f"Unsupported Pinocchio integrator: {method!r}")

        self.time += time_step
        self.forward()

    def _step_semi_implicit(self, dt: float) -> None:
        """Advance with velocity-first symplectic Euler integration."""
        if self.model is None or self.data is None:
            return

        self.a = pin.aba(self.model, self.data, self.q, self.v, self.tau)
        self.v = self.v + self.a * dt
        self.q = pin.integrate(self.model, self.q, self.v * dt)

    def _step_rk4(self, dt: float) -> None:
        """Advance with fourth-order Runge-Kutta on Pinocchio tangent state."""
        if self.model is None or self.data is None:
            return

        q0 = self.q.copy()
        v0 = self.v.copy()
        tau = self.tau.copy()

        def acceleration(q: np.ndarray, v: np.ndarray) -> np.ndarray:
            return cast(np.ndarray, pin.aba(self.model, self.data, q, v, tau))

        a1 = acceleration(q0, v0)
        v2 = v0 + 0.5 * dt * a1
        q2 = pin.integrate(self.model, q0, 0.5 * dt * v0)

        a2 = acceleration(q2, v2)
        v3 = v0 + 0.5 * dt * a2
        q3 = pin.integrate(self.model, q0, 0.5 * dt * v2)

        a3 = acceleration(q3, v3)
        v4 = v0 + dt * a3
        q4 = pin.integrate(self.model, q0, dt * v3)

        a4 = acceleration(q4, v4)

        weighted_velocity = (v0 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0
        weighted_acceleration = (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0

        self.q = pin.integrate(self.model, q0, dt * weighted_velocity)
        self.v = v0 + dt * weighted_acceleration
        self.a = a4.copy()

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
        """Set the current state and refresh derived kinematics."""
        if not (q is not None):
            raise ValueError("q must be provided")
        if not (q is not None):
            raise ValueError("q must be provided")
        if self.model is None:
            return

        if len(q) != self.model.nq:
            raise ValueError(f"q size {len(q)} does not match model nq={self.model.nq}")
        if len(v) != self.model.nv:
            raise ValueError(f"v size {len(v)} does not match model nv={self.model.nv}")
        self.q = q.copy()
        self.v = v.copy()
        # Refresh derived kinematics so Jacobians and frame placements are current
        self.forward()

    def set_control(self, u: np.ndarray) -> None:
        """Apply control inputs (torques/forces)."""
        if not (u is not None):
            raise ValueError("u must be provided")

        if self.model is None:
            return

        u_arr = self._require_vector("u", u, self.model.nv)
        self.tau = u_arr.copy()

    @precondition(lambda self, qacc: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Inverse dynamics must contain finite values")
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute torques required to achieve a given joint acceleration.

        Args:
            qacc: Desired joint acceleration vector (nv,)

        Returns:
            tau: Required joint torque vector (nv,)
        """
        if not (qacc is not None):
            raise ValueError("qacc must be provided")

        if self.model is None or self.data is None:
            return np.array([])

        qacc_arr = self._require_vector("qacc", qacc, self.model.nv)
        tau = pin.rnea(self.model, self.data, self.q, self.v, qacc_arr)
        return cast(np.ndarray, tau).copy()

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(
        lambda result: result is not None and result.shape == (3,),
        "Contact forces must be a (3,) array",
    )
    def compute_contact_forces(self) -> np.ndarray:
        """Compute total contact forces (ground reaction force, GRF).

        Pinocchio's standard ABA dynamics do not compute contact forces
        without a constraint solver (e.g., RigidContactModel + ProximalContactSolver).
        This implementation returns a zero-force fallback to allow callers
        to degrade gracefully to static gravity approximations.

        For accurate contact-aware dynamics, use Drake, MuJoCo, or a
        constraint-enabled Pinocchio configuration.

        Returns:
            Zero force vector [N] (3,) as fallback for unsupported contact queries.
        """
        if self.model is None or self.data is None:
            return np.array([0.0, 0.0, 0.0])

        # Return zero vector; callers check norm and fall back to gravity
        return np.array([0.0, 0.0, 0.0])

    @precondition(
        lambda self, body_name: self.is_initialized,
        "Engine must be initialized",
    )
    @postcondition(
        lambda res: res is None or all(check_finite(v) for v in res.values()),
        "Jacobian matrices must contain finite values",
    )
    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body."""
        if not (body_name is not None):
            raise ValueError("body_name must be provided")
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
        if not (tau is not None):
            raise ValueError("tau must be provided")
        if self.model is None or self.data is None:
            return np.array([])

        q_arr = self._require_vector("q", self.q, self.model.nq)
        v_arr = self._require_vector("v", self.v, self.model.nv)

        tau_zero = np.zeros(self.model.nv)
        a_drift = pin.aba(self.model, self.data, q_arr, v_arr, tau_zero)
        a_full = pin.aba(self.model, self.data, q_arr, v_arr, tau)

        return cast(np.ndarray, a_full - a_drift)

    @precondition(lambda self, q: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "ZVCF acceleration must contain finite values")
    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compute acceleration with joint velocities set to zero.

        Args:
            q: Joint positions (n_q,) [rad or m]

        Returns:
            q_ddot_ZVCF: Acceleration with v=0 (n_v,)
        """

        if not (q is not None):
            raise ValueError("q must be provided")
        if self.model is None or self.data is None:
            return np.array([])

        q_arr = self._require_vector("q", q, self.model.nq)

        v_zero = np.zeros(self.model.nv)

        # Use current control (preserved for ZVCF)
        tau = self.tau.copy()

        a_zvcf = pin.aba(self.model, self.data, q_arr, v_zero, tau)

        return cast(np.ndarray, a_zvcf)

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Affine drift must contain finite values")
    def compute_affine_drift(self) -> np.ndarray:
        """Compute the 'Drift' vector f(q, qdot).

        Legacy method - use compute_drift_acceleration() for Section F compliance.

        Returns acceleration when tau = 0 (and no active control).
        """
        return self.compute_drift_acceleration()

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    def get_sensors(self) -> dict[str, float | np.ndarray]:
        """Get all sensor readings.

        Currently Pinocchio standard data doesn't map directly to sensor definitions.
        Returns empty dict for parity with unconfigured MuJoCo models.
        """
        return {}

    # -------- PhysicsEngine protocol implementations (CRBA-based) --------

    @postcondition(check_finite, "Mass matrix must contain finite values")
    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the joint-space mass matrix M(q) via CRBA.

        Returns an empty array when no model has been loaded so callers can
        degrade gracefully.
        """
        if self.model is None or self.data is None:
            return np.array([])
        M = pin.crba(self.model, self.data, self.q)
        # CRBA fills only the upper triangle; symmetrise for downstream callers.
        M_arr = np.asarray(M)
        M_full = np.triu(M_arr) + np.triu(M_arr, 1).T
        return cast(np.ndarray, M_full)

    @postcondition(check_finite, "Bias forces must contain finite values")
    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v)v + g(q) via RNEA with zero acceleration."""
        if self.model is None or self.data is None:
            return np.array([])
        zero_acc = np.zeros(self.model.nv)
        nle = pin.rnea(self.model, self.data, self.q, self.v, zero_acc)
        return cast(np.ndarray, np.asarray(nle).copy())

    @postcondition(check_finite, "Gravity forces must contain finite values")
    def compute_gravity_forces(self) -> np.ndarray:
        """Compute generalised gravity forces g(q) via RNEA at v=a=0."""
        if self.model is None or self.data is None:
            return np.array([])
        zero_v = np.zeros(self.model.nv)
        zero_a = np.zeros(self.model.nv)
        g = pin.rnea(self.model, self.data, self.q, zero_v, zero_a)
        return cast(np.ndarray, np.asarray(g).copy())

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual: passive acceleration at (q, v).

        Restores the engine's stored (q, v) before returning so the engine
        state is untouched.
        """
        if not (q is not None):
            raise ValueError("q must be provided")
        if not (v is not None):
            raise ValueError("v must be provided")
        if self.model is None or self.data is None:
            return np.array([])
        q_arr = self._require_vector("q", q, self.model.nq)
        v_arr = self._require_vector("v", v, self.model.nv)
        tau_zero = np.zeros(self.model.nv)
        a_ztcf = pin.aba(self.model, self.data, q_arr, v_arr, tau_zero)
        return cast(np.ndarray, np.asarray(a_ztcf).copy())
