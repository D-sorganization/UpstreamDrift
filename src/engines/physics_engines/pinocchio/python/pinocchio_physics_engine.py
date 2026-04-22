"""Pinocchio Physics Engine wrapper implementation.

Wraps pinocchio to provide a compliant PhysicsEngine interface.

Inherits from BasePhysicsEngine to eliminate DRY violations for checkpoint
save/restore, model loading boilerplate, model name tracking, and
initialization patterns.
"""

from __future__ import annotations

from typing import Any, Literal, cast

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


def _require_vector_shape(
    name: str, value: np.ndarray, expected_size: int
) -> np.ndarray:
    """Return a one-dimensional vector or raise a dimension-specific error."""
    if value is None:
        raise ValueError(f"{name} must be provided")

    vector = np.asarray(value)
    if vector.ndim != 1 or vector.shape[0] != expected_size:
        actual_shape = tuple(vector.shape)
        raise ValueError(
            f"{name} expected 1-D size {expected_size}; actual shape {actual_shape}"
        )
    return vector


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
        lambda self, dt=None, integrator="semi_implicit": self.is_initialized,
        "Engine must be initialized",
    )
    def step(
        self,
        dt: float | None = None,
        integrator: Literal["semi_implicit", "rk4"] = "semi_implicit",
    ) -> None:
        """Advance the simulation by one time step.

        Args:
            dt: Time step size [s]. Defaults to DEFAULT_TIME_STEP.
            integrator: Integration scheme — ``"semi_implicit"`` (symplectic
                Euler, O(dt), energy-stable) or ``"rk4"`` (classic 4th-order
                Runge-Kutta, O(dt^4), more accurate for large dt or
                validation).
        """
        if self.model is None or self.data is None:
            return

        time_step = dt if dt is not None else DEFAULT_TIME_STEP

        if integrator == "rk4":
            self._step_rk4(time_step)
        else:
            self._step_semi_implicit(time_step)

        self.time += time_step

    def _step_semi_implicit(self, time_step: float) -> None:
        """Symplectic (semi-implicit) Euler: velocity-first, then position."""
        self.a = pin.aba(self.model, self.data, self.q, self.v, self.tau)
        self.v += self.a * time_step
        self.q = pin.integrate(self.model, self.q, self.v * time_step)

    def _step_rk4(self, time_step: float) -> None:
        """Classic RK4 integration over Pinocchio's Lie-group configuration."""
        q0, v0 = self.q.copy(), self.v.copy()
        tau = self.tau

        def dv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
            return pin.aba(self.model, self.data, q, v, tau).copy()

        # k1
        a1 = dv(q0, v0)
        # k2
        v_k2 = v0 + 0.5 * time_step * a1
        q_k2 = pin.integrate(self.model, q0, 0.5 * time_step * v0)
        a2 = dv(q_k2, v_k2)
        # k3
        v_k3 = v0 + 0.5 * time_step * a2
        q_k3 = pin.integrate(self.model, q0, 0.5 * time_step * v_k2)
        a3 = dv(q_k3, v_k3)
        # k4
        v_k4 = v0 + time_step * a3
        q_k4 = pin.integrate(self.model, q0, time_step * v_k3)
        a4 = dv(q_k4, v_k4)

        # Weighted update — velocity in R^n, position on Lie group
        dv_weighted = (time_step / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
        dq_weighted = (time_step / 6.0) * (v0 + 2 * v_k2 + 2 * v_k3 + v_k4)
        self.v = v0 + dv_weighted
        self.q = pin.integrate(self.model, q0, dq_weighted)
        self.a = a4

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
        if q is None:
            raise ValueError("q must be provided")
        if v is None:
            raise ValueError("v must be provided")
        if self.model is None:
            return

        q_vector = _require_vector_shape("q", q, self.model.nq)
        v_vector = _require_vector_shape("v", v, self.model.nv)
        self.q = q_vector.copy()
        self.v = v_vector.copy()
        # Refresh derived kinematics so Jacobians and frame placements are current
        self.forward()

    def set_control(self, u: np.ndarray) -> None:
        """Apply control inputs (torques/forces)."""
        if u is None:
            raise ValueError("u must be provided")
        if self.model is None:
            return

        control = _require_vector_shape("u", u, self.model.nv)
        self.tau = control.copy()

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
        if not (qacc is not None):
            raise ValueError("qacc must be provided")
        if not (qacc is not None):
            raise ValueError("qacc must be provided")
        if self.model is None or self.data is None:
            return np.array([])

        tau = pin.rnea(self.model, self.data, self.q, self.v, qacc)
        return cast(np.ndarray, tau)

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Contact forces must contain finite values")
    def compute_contact_forces(self) -> np.ndarray:
        """Compute total contact forces (ground reaction force, GRF).

        Notes:
            Pinocchio's standard ABA does not compute contact forces
            without a constraint solver. Return zeros to maintain parity
            with cross-engine protocol until rigidly implemented.

        Returns:
            f: (3,) vector representing total ground reaction force [N].
        """
        logger.debug(
            "PinocchioPhysicsEngine.compute_contact_forces is returning zeros. "
            "Standard ABA dynamics in Pinocchio do not compute contact forces "
            "without a constraint solver."
        )
        return np.zeros(3)

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

        tau_vector = _require_vector_shape("tau", tau, self.model.nv)

        M = self.compute_mass_matrix()
        if M.size == 0:
            return np.array([])

        a_control = np.linalg.solve(M, tau_vector)

        return a_control

    @precondition(lambda self, q, v: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "ZTCF acceleration must contain finite values")
    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compute acceleration with applied torques set to zero.

        Args:
            q: Joint positions (n_q,) [rad or m]
            v: Joint velocities (n_v,) [rad/s or m/s]

        Returns:
            q_ddot_ZTCF: Acceleration under zero torque (n_v,)
        """
        if not (q is not None):
            raise ValueError("q must be provided")
        if self.model is None or self.data is None:
            return np.array([])

        q_vector = _require_vector_shape("q", q, self.model.nq)
        v_vector = _require_vector_shape("v", v, self.model.nv)

        tau_zero = np.zeros(self.model.nv)
        a_ztcf = pin.aba(self.model, self.data, q_vector, v_vector, tau_zero)

        return cast(np.ndarray, a_ztcf)

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

        q_vector = _require_vector_shape("q", q, self.model.nq)

        v_zero = np.zeros(self.model.nv)

        # Use current control (preserved for ZVCF)
        tau = self.tau.copy()

        a_zvcf = pin.aba(self.model, self.data, q_vector, v_zero, tau)

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
