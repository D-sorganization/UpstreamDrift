"""Double Pendulum Physics Engine Adapter.

Wraps the standalone DoublePendulumDynamics to implement the PhysicsEngine
protocol. Inherits from BasePhysicsEngine to eliminate DRY violations for
checkpoint save/restore, model name tracking, and engine initialization.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumState,
)
from src.shared.python.core.contracts import (
    check_finite,
    postcondition,
    precondition,
)
from src.shared.python.engine_core.base_physics_engine import (
    BasePhysicsEngine,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class PendulumPhysicsEngine(BasePhysicsEngine):
    """Adapter for DoublePendulumDynamics matching PhysicsEngine protocol.

    Inherits common functionality from BasePhysicsEngine:
    - Checkpoint save/restore (with phi/omega_phi via hooks)
    - Model name and initialization tracking
    - String representation
    """

    def __init__(self) -> None:
        """Initialize the pendulum engine."""
        super().__init__()
        self.dynamics = DoublePendulumDynamics()
        # Initial state
        self._pendulum_state = DoublePendulumState(
            theta1=0.0,
            theta2=0.0,
            omega1=0.0,
            omega2=0.0,
            phi=0.0,
            omega_phi=0.0,
        )
        self.time = 0.0
        self.control = np.zeros(2)

        # Wire up forcing functions to use our control
        self.dynamics.forcing_functions = (
            self._get_shoulder_torque,
            self._get_wrist_torque,
        )

        # Pendulum is a fixed model: always initialized
        self.model = self.dynamics
        self.model_name_str = "DoublePendulum"
        self._is_initialized = True

    def _get_shoulder_torque(self, t: float, state: DoublePendulumState) -> float:
        return float(self.control[0])

    def _get_wrist_torque(self, t: float, state: DoublePendulumState) -> float:
        return float(self.control[1])

    @property
    def engine_type(self) -> str:
        """Get engine type identifier."""
        return "pendulum"

    def _load_from_path_impl(self, path: str) -> None:
        """Engine-specific load from path (no-op for pendulum)."""

    def _load_from_string_impl(self, content: str, extension: str | None) -> None:
        """Engine-specific load from string (no-op for pendulum)."""

    def load_from_path(self, path: str) -> None:
        """Load model from file path.

        Pendulum is a standalone fixed model. Path is ignored.
        Overrides base to skip path validation (no actual file).
        """
        logger.debug(
            "PendulumPhysicsEngine is standalone. "
            "Model parameters are default. Path %s ignored.",
            path,
        )

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from string.

        Pendulum is a standalone fixed model. Content is ignored.
        Overrides base to skip content validation.
        """
        logger.debug("PendulumPhysicsEngine ignores load_from_string.")

    def reset(self) -> None:
        """Reset simulation state to initial configuration."""
        self._pendulum_state = DoublePendulumState(
            theta1=0.0,
            theta2=0.0,
            omega1=0.0,
            omega2=0.0,
            phi=0.0,
            omega_phi=0.0,
        )
        self.time = 0.0
        self.control = np.zeros(2)

    def step(self, dt: float | None = None) -> None:
        """Step the simulation forward."""
        step_size = dt if dt is not None else 0.01

        # The dynamics step returns a NEW state object (functional style)
        self._pendulum_state = self.dynamics.step(
            self.time, self._pendulum_state, step_size
        )
        self.time += step_size

    def forward(self) -> None:
        """Compute forward kinematics/dynamics without stepping time."""
        # Pendulum dynamics are computed on-the-fly in step or accessors.

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities)."""
        q = np.array([self._pendulum_state.theta1, self._pendulum_state.theta2])
        v = np.array([self._pendulum_state.omega1, self._pendulum_state.omega2])
        return q, v

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state."""
        if len(q) >= 2 and len(v) >= 2:
            self._pendulum_state.theta1 = float(q[0])
            self._pendulum_state.theta2 = float(q[1])
            self._pendulum_state.omega1 = float(v[0])
            self._pendulum_state.omega2 = float(v[1])

    def set_control(self, u: np.ndarray) -> None:
        """Set control vector."""
        if len(u) >= 2:
            self.control = u.copy()

    def get_time(self) -> float:
        """Get the current simulation time."""
        return self.time

    # -- Checkpoint Hooks (DRY: delegates to BasePhysicsEngine) --

    def _get_extra_checkpoint_state(self) -> dict[str, Any]:
        """Return pendulum-specific checkpoint data (phi, omega_phi)."""
        return {
            "phi": self._pendulum_state.phi,
            "omega_phi": self._pendulum_state.omega_phi,
        }

    def _restore_extra_checkpoint_state(self, checkpoint: StateCheckpoint) -> None:
        """Restore pendulum-specific state from checkpoint."""
        self.time = checkpoint.timestamp
        if "phi" in checkpoint.engine_state:
            self._pendulum_state.phi = checkpoint.engine_state["phi"]
        if "omega_phi" in checkpoint.engine_state:
            self._pendulum_state.omega_phi = checkpoint.engine_state["omega_phi"]

    # -------- Dynamics Interface --------

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Mass matrix must contain finite values")
    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q)."""
        m_tuple = self.dynamics.mass_matrix(self._pendulum_state.theta2)
        return np.array(m_tuple)

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Bias forces must contain finite values")
    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q,v) + g(q) + d(q,v)."""
        c1, c2 = self.dynamics.coriolis_vector(
            self._pendulum_state.theta2,
            self._pendulum_state.omega1,
            self._pendulum_state.omega2,
        )
        g1, g2 = self.dynamics.gravity_vector(
            self._pendulum_state.theta1, self._pendulum_state.theta2
        )
        d1, d2 = self.dynamics.damping_vector(
            self._pendulum_state.omega1, self._pendulum_state.omega2
        )

        return np.array([c1 + g1 + d1, c2 + g2 + d2])

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Gravity forces must contain finite values")
    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q)."""
        g1, g2 = self.dynamics.gravity_vector(
            self._pendulum_state.theta1, self._pendulum_state.theta2
        )
        return np.array([g1, g2])

    @precondition(
        lambda self, qacc: self.is_initialized,
        "Engine must be initialized",
    )
    @postcondition(check_finite, "Inverse dynamics torques must contain finite values")
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics tau = ID(q, v, a)."""
        if len(qacc) < 2:
            return np.array([])

        tau1, tau2 = self.dynamics.inverse_dynamics(
            self._pendulum_state, (float(qacc[0]), float(qacc[1]))
        )
        return np.array([tau1, tau2])

    @precondition(lambda self: self.is_initialized, "Engine must be initialized")
    @postcondition(check_finite, "Drift acceleration must contain finite values")
    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (drift) acceleration with zero control inputs.

        Section F Implementation: Returns acceleration with tau=0.
        """
        M = self.compute_mass_matrix()
        bias = self.compute_bias_forces()

        # M*a_drift + bias = 0 => a_drift = -M^-1 * bias
        a_drift = np.linalg.solve(M, -bias)
        return a_drift

    @precondition(
        lambda self, tau: self.is_initialized,
        "Engine must be initialized",
    )
    @postcondition(check_finite, "Control acceleration must contain finite values")
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Compute control-attributed acceleration from applied torques.

        Section F Implementation: Returns M^-1 * tau.

        Args:
            tau: Applied generalized forces (2,) [N*m]

        Returns:
            Control acceleration vector (2,) [rad/s**2]
        """
        if len(tau) < 2:
            return np.array([])

        M = self.compute_mass_matrix()
        a_control = np.linalg.solve(M, tau)
        return a_control

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body."""
        # Placeholder -- not yet implemented for double pendulum
        return None

    # ---- Section G: Counterfactual Experiments (Implementation) ----

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual - Guideline G1.

        For double pendulum:
            q_ddot_ZTCF = M(q)^-1 * (-(C(q,v)*v + g(q) + d(q,v)))

        Args:
            q: Joint positions [rad] (2,)
            v: Joint velocities [rad/s] (2,)

        Returns:
            Acceleration with tau=0 [rad/s**2] (2,)
        """
        if len(q) < 2 or len(v) < 2:
            return np.array([])

        # Save current state
        theta1_orig = self._pendulum_state.theta1
        theta2_orig = self._pendulum_state.theta2
        omega1_orig = self._pendulum_state.omega1
        omega2_orig = self._pendulum_state.omega2

        try:
            # Set to counterfactual state
            self._pendulum_state.theta1 = float(q[0])
            self._pendulum_state.theta2 = float(q[1])
            self._pendulum_state.omega1 = float(v[0])
            self._pendulum_state.omega2 = float(v[1])

            a_ztcf = self.compute_drift_acceleration()
            return a_ztcf

        finally:
            # Restore original state
            self._pendulum_state.theta1 = theta1_orig
            self._pendulum_state.theta2 = theta2_orig
            self._pendulum_state.omega1 = omega1_orig
            self._pendulum_state.omega2 = omega2_orig

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual - Guideline G2.

        For double pendulum with v=0:
            q_ddot_ZVCF = M(q)^-1 * (-g(q) + tau)

        Args:
            q: Joint positions [rad] (2,)

        Returns:
            Acceleration with v=0 but tau preserved [rad/s**2] (2,)
        """
        if len(q) < 2:
            return np.array([])

        # Save current state
        theta1_orig = self._pendulum_state.theta1
        theta2_orig = self._pendulum_state.theta2
        omega1_orig = self._pendulum_state.omega1
        omega2_orig = self._pendulum_state.omega2

        try:
            self._pendulum_state.theta1 = float(q[0])
            self._pendulum_state.theta2 = float(q[1])
            self._pendulum_state.omega1 = 0.0  # ZVCF: zero velocity
            self._pendulum_state.omega2 = 0.0

            g1, g2 = self.dynamics.gravity_vector(
                self._pendulum_state.theta1,
                self._pendulum_state.theta2,
            )
            g = np.array([g1, g2])

            tau = self.control.copy()

            M = self.compute_mass_matrix()
            a_zvcf = np.linalg.solve(M, -g + tau)

            return a_zvcf

        finally:
            self._pendulum_state.theta1 = theta1_orig
            self._pendulum_state.theta2 = theta2_orig
            self._pendulum_state.omega1 = omega1_orig
            self._pendulum_state.omega2 = omega2_orig
