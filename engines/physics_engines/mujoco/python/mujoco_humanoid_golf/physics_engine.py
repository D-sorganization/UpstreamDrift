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
