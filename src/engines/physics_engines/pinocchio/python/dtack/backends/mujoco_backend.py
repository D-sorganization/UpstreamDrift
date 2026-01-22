"""MuJoCo backend wrapper for simulation and contact dynamics."""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np  # noqa: TID253
import numpy.typing as npt  # noqa: TID253

logger = logging.getLogger(__name__)


class MuJoCoBackend:
    """MuJoCo backend for forward simulation and contact dynamics.

    This backend provides:
    - MJCF model loading
    - Forward simulation with contacts
    - Contact force computation
    - Sensor data access
    - Forward/inverse dynamics via MuJoCo
    """

    def __init__(self, model_path: Path | str) -> None:
        """Initialize MuJoCo backend.

        Args:
            model_path: Path to MJCF file or canonical YAML specification

        Raises:
            FileNotFoundError: If model file does not exist
            ValueError: If model cannot be loaded
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # For now, assume MJCF XML. Later we'll add YAML parser
        if model_path_obj.suffix in (".xml", ".mjcf"):
            self.model = mujoco.MjModel.from_xml_path(str(model_path_obj))
        else:
            msg = f"Unsupported model format: {model_path_obj.suffix}"
            raise ValueError(msg)

        self.data = mujoco.MjData(self.model)

        logger.info(
            "Loaded MuJoCo model: %d DOF, %d bodies",
            self.model.nv,
            self.model.nbody,
        )

    def step(self, ctrl: npt.NDArray[np.float64] | None = None) -> None:
        """Step simulation forward.

        Args:
            ctrl: Control input [nu]. If None, uses current data.ctrl
        """
        if ctrl is not None:
            self.data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
        mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        """Compute forward dynamics without stepping."""
        mujoco.mj_forward(self.model, self.data)

    def get_qpos(self) -> npt.NDArray[np.float64]:
        """Get joint positions.

        Returns:
            Joint positions [nq]
        """
        result = self.data.qpos.copy()
        return np.asarray(result, dtype=np.float64)

    def get_qvel(self) -> npt.NDArray[np.float64]:
        """Get joint velocities.

        Returns:
            Joint velocities [nv]
        """
        result = self.data.qvel.copy()
        return np.asarray(result, dtype=np.float64)

    def get_qacc(self) -> npt.NDArray[np.float64]:
        """Get joint accelerations.

        Returns:
            Joint accelerations [nv]
        """
        result = self.data.qacc.copy()
        return np.asarray(result, dtype=np.float64)

    def set_qpos(self, q: npt.NDArray[np.float64]) -> None:
        """Set joint positions.

        Args:
            q: Joint positions [nq]
        """
        self.data.qpos[:] = np.asarray(q, dtype=np.float64)

    def set_qvel(self, v: npt.NDArray[np.float64]) -> None:
        """Set joint velocities.

        Args:
            v: Joint velocities [nv]
        """
        self.data.qvel[:] = np.asarray(v, dtype=np.float64)

    def compute_inverse_dynamics(
        self,
        q: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        a: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute inverse dynamics.

        Args:
            q: Joint positions [nq]
            v: Joint velocities [nv]
            a: Joint accelerations [nv]

        Returns:
            Joint torques [nv]
        """
        q_arr = np.asarray(q, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        a_arr = np.asarray(a, dtype=np.float64)

        self.set_qpos(q_arr)
        self.set_qvel(v_arr)
        mujoco.mj_inverse(self.model, self.data)
        self.data.qacc[:] = a_arr
        mujoco.mj_forward(self.model, self.data)
        result = self.data.qfrc_inverse.copy()
        return np.asarray(result, dtype=np.float64)

    def get_contact_forces(self) -> npt.NDArray[np.float64]:
        """Get contact forces.

        Returns:
            Contact forces [ncon * 6] (force + torque for each contact)
        """
        forces = np.zeros(self.data.ncon * 6)
        for i in range(self.data.ncon):
            mujoco.mj_contactForce(
                self.model, self.data, i, forces[i * 6 : (i + 1) * 6]
            )
        return forces
