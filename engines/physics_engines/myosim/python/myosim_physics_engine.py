"""MyoSim Physics Engine Implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, cast

import numpy as np

from shared.python.interfaces import PhysicsEngine

# Configure logging
logger = logging.getLogger(__name__)

try:
    import mujoco
except ImportError:
    mujoco = None
    logger.warning("MuJoCo python package not found. MyoSimPhysicsEngine will not function.")


class MyoSimPhysicsEngine(PhysicsEngine):
    """MyoSim Physics Engine Implementation (MuJoCo-based)."""

    def __init__(self) -> None:
        """Initialize the physics engine."""
        self.model: Any | None = None
        self.data: Any | None = None
        self.xml_path: str | None = None
        
        if mujoco is None:
            logger.error("MuJoCo library is not installed.")

    @property
    def model_name(self) -> str:
        """Return the name of the currently loaded model."""
        if self.model is None:
            return "MyoSim_NoModel"
        return "MyoSim Model"

    def load_from_path(self, path: str) -> None:
        """Load model from file path."""
        if mujoco is None:
            raise ImportError("MuJoCo library not installed")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(path)
            self.data = mujoco.MjData(self.model)
            self.xml_path = path
            logger.info(f"Loaded MyoSim model from {path}")
        except Exception as e:
            logger.error(f"Failed to load MyoSim model: {e}")
            raise

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load model from XML string."""
        if mujoco is None:
            raise ImportError("MuJoCo library not installed")

        try:
            self.model = mujoco.MjModel.from_xml_string(content)
            self.data = mujoco.MjData(self.model)
            self.xml_path = None
        except Exception as e:
            logger.error(f"Failed to load MyoSim model from string: {e}")
            raise

    def reset(self) -> None:
        """Reset simulation state to initial configuration."""
        if self.model is not None and self.data is not None:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

    def step(self, dt: float | None = None) -> None:
        """Step the simulation forward."""
        if self.model is not None and self.data is not None:
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

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the current state (positions, velocities)."""
        if self.data is None:
            return np.array([]), np.array([])
        return self.data.qpos.copy(), self.data.qvel.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the current state."""
        if self.data is not None:
            if len(q) == len(self.data.qpos):
                self.data.qpos[:] = q
            if len(v) == len(self.data.qvel):
                self.data.qvel[:] = v
            mujoco.mj_forward(self.model, self.data)

    def set_control(self, u: np.ndarray) -> None:
        """Set control vector."""
        if self.data is not None and self.model is not None:
            if len(u) == self.model.nu:
                self.data.ctrl[:] = u
            else:
                logger.warning(
                    f"Control vector size mismatch: got {len(u)}, expected {self.model.nu}"
                )

    def get_time(self) -> float:
        """Get the current simulation time."""
        if self.data is None:
            return 0.0
        return float(self.data.time)

    def compute_mass_matrix(self) -> np.ndarray:
        """Compute the dense inertia matrix M(q)."""
        if self.model is None or self.data is None:
            return np.array([])

        nv = self.model.nv
        M = np.zeros((nv, nv), dtype=np.float64)

        if hasattr(mujoco, "mj_makeInertia"):
            mujoco.mj_makeInertia(self.model, self.data)

        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def compute_bias_forces(self) -> np.ndarray:
        """Compute bias forces C(q, qdot) + g(q)."""
        if self.data is None:
            return np.array([])
        # Populated after mj_step or mj_forward
        return cast(np.ndarray, self.data.qfrc_bias.copy())

    def compute_gravity_forces(self) -> np.ndarray:
        """Compute gravity forces g(q)."""
        # MuJoCo doesn't isolate gravity easily without modification or using qfrc_grav (if available)
        # qfrc_grav is available in newer MuJoCo versions
        if self.data is None:
            return np.array([])
            
        if hasattr(self.data, "qfrc_grav"):
             return cast(np.ndarray, self.data.qfrc_grav.copy())
             
        return np.array([])

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Compute inverse dynamics."""
        if self.model is None or self.data is None:
            return np.array([])
            
        if len(qacc) != self.model.nv:
            return np.array([])
            
        self.data.qacc[:] = qacc
        mujoco.mj_inverse(self.model, self.data)
        return cast(np.ndarray, self.data.qfrc_inverse.copy())

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
            "spatial": np.vstack([jacp, jacr]),
        }
