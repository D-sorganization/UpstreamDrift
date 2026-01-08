from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, cast  # noqa: F401

import mujoco
import numpy as np
from shared.python.interfaces import PhysicsEngine
from shared.python.security_utils import validate_path

LOGGER = logging.getLogger(__name__)

# Model directories allowed for loading (relative to suite root)
# Hardening: Prevent loading from arbitrary system paths
SUITE_ROOT = Path(__file__).parents[5]
ALLOWED_MODEL_DIRS = [
    SUITE_ROOT / "engines",
    SUITE_ROOT / "shared" / "resources",
]


class MuJoCoPhysicsEngine(PhysicsEngine):
    """Encapsulates MuJoCo model, data, and simulation control.

    Implements the shared PhysicsEngine protocol.
    """

    def __init__(self) -> None:
        """Initialize the physics engine."""
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.xml_path: str | None = None

    @property
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        if self.model is None:
            return "None"

        # Try to get from model names
        if self.model.names:
            # The names buffer is a byte string, need to handle encoding
            # if accessed directly
            # But typically we want the model name from the XML
            return "MuJoCo Model"

        if self.xml_path:
            return os.path.basename(self.xml_path).replace(".xml", "")

        return "MuJoCo Model"

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from XML string."""
        try:
            self.model = mujoco.MjModel.from_xml_string(content)
            self.data = mujoco.MjData(self.model)
            self.xml_path = None
        except Exception as e:
            LOGGER.error("Failed to load model from XML string: %s", e)
            raise

    def load_from_path(self, path: str) -> None:
        """Load model from file path."""
        try:
            # Security: Validate path is within allowed directories
            # Hardening against path traversal (F-004)
            resolved = validate_path(path, ALLOWED_MODEL_DIRS, strict=True)
            path_str = str(resolved)

            self.model = mujoco.MjModel.from_xml_path(path_str)
            self.data = mujoco.MjData(self.model)
            self.xml_path = path_str
        except Exception as e:
            LOGGER.error("Failed to load model from path %s: %s", path, e)
            raise

    def set_model_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Set model and data manually (e.g. from async loader)."""
        self.model = model
        self.data = data
        self.xml_path = None

    def get_model(self) -> mujoco.MjModel | None:
        return self.model

    def get_data(self) -> mujoco.MjData | None:
        return self.data

    def step(self, dt: float | None = None) -> None:
        """Step the simulation forward."""
        if self.model is not None and self.data is not None:
            # If dt is provided, temporarily override option.timestep
            if dt is not None:
                old_dt = self.model.opt.timestep
                self.model.opt.timestep = dt
                mujoco.mj_step(self.model, self.data)
                self.model.opt.timestep = old_dt
            else:
                mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        """Compute forward kinematics/dynamics without stepping time."""
        if self.model is not None and self.data is not None:
            mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Reset simulation state to initial configuration."""
        if self.model is not None and self.data is not None:
            mujoco.mj_resetData(self.model, self.data)
            self.forward()

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities)."""
        if self.data is None:
            return np.array([]), np.array([])
        return self.data.qpos.copy(), self.data.qvel.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state."""
        if self.data is not None:
            # Validate dimensions
            if len(q) != len(self.data.qpos):
                raise ValueError(
                    f"State q size mismatch: got {len(q)}, "
                    f"expected {len(self.data.qpos)}"
                )
            self.data.qpos[:] = q

            if len(v) != len(self.data.qvel):
                raise ValueError(
                    f"State v size mismatch: got {len(v)}, "
                    f"expected {len(self.data.qvel)}"
                )
            self.data.qvel[:] = v

            # Critical: Update derived quantities (accelerations, sensors, etc.)
            self.forward()

    def set_control(self, u: np.ndarray) -> None:
        """Set control vector."""
        if self.data is not None and self.model is not None:
            # Strict size validation
            if len(u) != self.model.nu:
                raise ValueError(
                    f"Control vector size mismatch: got {len(u)}, "
                    f"expected {self.model.nu}"
                )
            self.data.ctrl[:] = u

    def get_time(self) -> float:
        """Get the current simulation time."""
        if self.data is None:
            return 0.0
        return float(self.data.time)

    # -------- Section 1: Core Dynamics Engine Capabilities --------

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        nv = self.model.nv
        M = np.zeros((nv, nv), dtype=np.float64)

        # Ensure qM is updated
        if hasattr(mujoco, "mj_makeInertia"):
            mujoco.mj_makeInertia(self.model, self.data)

        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q, qdot) + g(q)."""
        if self.data is None:
            return np.array([])
        # This is populated after mj_forward/mj_step
        return cast(np.ndarray, self.data.qfrc_bias.copy())

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q)."""
        if self.model is None or self.data is None:
            return np.array([])
        qfrc_grav = getattr(self.data, "qfrc_grav", None)
        if qfrc_grav is None:
            qfrc_grav = getattr(self.data, "qfrc_bias", np.zeros(self.model.nv))
        # Cast to ndarray before copy to satisfy mypy
        grav_arr = cast(np.ndarray, qfrc_grav)
        return grav_arr.copy()

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics: tau = ID(q, qdot, qacc)."""
        if self.model is None or self.data is None:
            return np.array([])

        if len(qacc) != self.model.nv:
            LOGGER.error("Dimension mismatch for qacc")
            return np.array([])

        # Copy qacc to data
        self.data.qacc[:] = qacc

        # Compute inverse dynamics
        mujoco.mj_inverse(self.model, self.data)

        return cast(np.ndarray, self.data.qfrc_inverse.copy())

    # -------- Section F: Drift-Control Decomposition --------

    def compute_drift_acceleration(self) -> np.ndarray:
        """Compute passive (drift) acceleration with zero control inputs.

        Section F Implementation: Uses MuJoCo's mj_forward with zero control
        to compute passive dynamics due to gravity and Coriolis/centrifugal forces.

        Returns:
            q_ddot_drift: Drift acceleration vector (nv,) [rad/s² or m/s²]
        """
        return self.compute_affine_drift()

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

        # Get mass matrix
        M = self.compute_mass_matrix()

        # Control component: M^-1 * tau
        a_control = np.linalg.solve(M, tau)

        return a_control

    def compute_affine_drift(self) -> np.ndarray:
        """Compute the 'Drift' vector f(q, qdot).

        Legacy method - use compute_drift_acceleration() for Section F compliance.

        Returns acceleration when tau = 0 (and no active control).
        """
        if self.model is None or self.data is None:
            return np.array([])

        # Save current control
        saved_ctrl = self.data.ctrl.copy()

        # Zero out control
        self.data.ctrl[:] = 0

        # Compute forward dynamics
        mujoco.mj_forward(self.model, self.data)
        drift_acc = self.data.qacc.copy()

        # Restore control
        self.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(self.model, self.data)  # Restore state

        return cast(np.ndarray, drift_acc)

    # -------- Section 4: Jacobian Analysis --------

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Compute spatial Jacobian for a specific body."""
        if self.model is None or self.data is None:
            return None

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return None

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

        return {
            "linear": jacp,
            "angular": jacr,
            "spatial": np.vstack([jacr, jacp]),
            # Standardized to [Angular; Linear] (Drake convention)
        }

    def get_sensors(self) -> dict[str, float]:
        """Get all sensor readings."""
        if self.data is None or self.model is None:
            return {}

        sensors = {}
        if self.model.nsensor > 0:
            for i in range(self.model.nsensor):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                if not name:
                    name = f"sensor_{i}"

                # Sensors can have dim > 1, but for simplicity handled as scalar or list
                # This is a basic implementation
                # Robust implementation would read adr and dim
                sensors[name] = float(self.data.sensordata[i])
        return sensors

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

        Compute acceleration with applied torques set to zero, preserving state.
        This isolates drift (gravity + Coriolis + constraints) from control effects.

        **Purpose**: Answer "What would happen if all actuators turned off RIGHT NOW?"

        **Physics**: With τ=0, acceleration is purely passive:
            q̈_ZTCF = M(q)⁻¹ · (C(q,v)·v + g(q) + J^T·λ)

        Args:
            q: Joint positions (n_q,) [rad or m]
            v: Joint velocities (n_v,) [rad/s or m/s]

        Returns:
            q̈_ZTCF: Acceleration under zero applied torque (n_v,) [rad/s² or m/s²]
        """
        if self.model is None or self.data is None:
            return np.array([])

        # Save current state and control
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_ctrl = self.data.ctrl.copy()

        try:
            # Set to counterfactual state
            self.data.qpos[:] = q
            self.data.qvel[:] = v

            # Zero out control (ZTCF: zero torque)
            self.data.ctrl[:] = 0

            # Compute forward dynamics with zero control
            mujoco.mj_forward(self.model, self.data)

            # Extract acceleration (this is the drift acceleration)
            a_ztcf = self.data.qacc.copy()

            return cast(np.ndarray, a_ztcf)

        finally:
            # Restore original state and control
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            self.data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.model, self.data)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

        Compute acceleration with joint velocities set to zero, preserving
        configuration.
        This isolates configuration-dependent effects (gravity, constraints)
        from velocity-dependent effects (Coriolis, centrifugal).

        **Purpose**: Answer "What acceleration would occur if motion FROZE
        instantaneously?"

        **Physics**: With v=0, acceleration has no velocity-dependent terms:
            q̈_ZVCF = M(q)⁻¹ · (g(q) + τ + J^T·λ)

        Args:
            q: Joint positions (n_q,) [rad or m]

        Returns:
            q̈_ZVCF: Acceleration with v=0 (n_v,) [rad/s² or m/s²]
        """
        if self.model is None or self.data is None:
            return np.array([])

        # Save current state
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()

        try:
            # Set to counterfactual configuration with v=0
            self.data.qpos[:] = q
            self.data.qvel[:] = 0  # ZVCF: zero velocity

            # Control is preserved from current state (already in data.ctrl)

            # Compute forward dynamics with zero velocity
            mujoco.mj_forward(self.model, self.data)

            # Extract acceleration
            a_zvcf = self.data.qacc.copy()

            return cast(np.ndarray, a_zvcf)

        finally:
            # Restore original state
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.model, self.data)
