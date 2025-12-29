"""MuJoCo Physics Engine encapsulation.

Manages the low-level MuJoCo simulation state, loading, and stepping.
"""

from __future__ import annotations

import logging
import os

import mujoco
import numpy as np

LOGGER = logging.getLogger(__name__)


class MuJoCoPhysicsEngine:
    """Encapsulates MuJoCo model, data, and simulation control."""

    def __init__(self) -> None:
        """Initialize the physics engine."""
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.xml_path: str | None = None

    def load_from_xml_string(self, xml_content: str) -> None:
        """Load model from XML string."""
        try:
            self.model = mujoco.MjModel.from_xml_string(xml_content)
            self.data = mujoco.MjData(self.model)
            self.xml_path = None
        except Exception as e:
            LOGGER.error("Failed to load model from XML string: %s", e)
            raise

    def load_from_path(self, xml_path: str) -> None:
        """Load model from file path."""
        try:
            # Convert to absolute path if needed
            if not os.path.isabs(xml_path):
                # Attempt to resolve relative to project root?
                # Or assume caller handles resolution.
                # For safety, let's assume absolute or relative to cwd.
                pass

            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            self.xml_path = xml_path
        except Exception as e:
            LOGGER.error("Failed to load model from path %s: %s", xml_path, e)
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

    def step(self) -> None:
        """Step the simulation forward."""
        if self.model is not None and self.data is not None:
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

    def set_control(self, ctrl: np.ndarray) -> None:
        """Set control vector."""
        if self.data is not None:
            # Ensure size matches
            if len(ctrl) == self.model.nu:
                self.data.ctrl[:] = ctrl
            else:
                LOGGER.warning(
                    "Control vector size mismatch: got %d, expected %d",
                    len(ctrl),
                    self.model.nu,
                )

    # -------- Section 1: Core Dynamics Engine Capabilities --------

    def compute_mass_matrix(self) -> np.ndarray | None:
        """Compute the dense inertia matrix M(q)."""
        if self.model is None or self.data is None:
            return None

        nv = self.model.nv
        M = np.zeros((nv, nv), dtype=np.float64)

        # Ensure qM is updated
        # Usually mj_forward or mj_step updates qM. But mj_fullM needs qM.
        # It's computed during inverse/forward dynamics.
        if hasattr(mujoco, "mj_makeInertia"):
            mujoco.mj_makeInertia(self.model, self.data)

        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def compute_bias_forces(self) -> np.ndarray | None:
        """Compute bias forces C(q, qdot) + g(q).

        Returns:
            Vector of size (nv,) containing Coriolis, Centrifugal, and Gravity forces.
        """
        if self.data is None:
            return None
        # This is populated after mj_forward/mj_step
        return self.data.qfrc_bias.copy()

    def compute_gravity_forces(self) -> np.ndarray | None:
        """Compute gravity forces g(q)."""
        if self.data is None:
            return None
        return self.data.qfrc_grav.copy()

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray | None:
        """Compute inverse dynamics: tau = ID(q, qdot, qacc).

        Computes the forces required to produce the given acceleration.
        Note: This writes to data.qfrc_inverse.
        """
        if self.model is None or self.data is None:
            return None

        if len(qacc) != self.model.nv:
            LOGGER.error("Dimension mismatch for qacc")
            return None

        # Copy qacc to data
        self.data.qacc[:] = qacc

        # Compute inverse dynamics
        mujoco.mj_inverse(self.model, self.data)

        return self.data.qfrc_inverse.copy()

    # -------- Section 3: Drift vs Control (Affine View) --------

    def compute_affine_drift(self) -> np.ndarray | None:
        """Compute the 'Drift' vector f(q, qdot).

        acceleration when tau = 0 (and no active control).
        """
        if self.model is None or self.data is None:
            return None

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

        return drift_acc

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
            "spatial": np.vstack([jacp, jacr]),
        }
