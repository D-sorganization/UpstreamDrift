import logging
import os

import mujoco

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MujocoModel:
    """
    Wrapper for MuJoCo simulation model and data.
    """

    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            logger.error(f"Failed to load MuJoCo model: {e}")
            raise

        self.paused = True
        self.reset()

    def reset(self) -> None:
        """Resets the simulation to the initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        logger.info("Simulation reset.")

    def step(self) -> None:
        """Steps the simulation forward by one timestep."""
        if not self.paused:
            mujoco.mj_step(self.model, self.data)

    def get_time(self) -> float:
        """Returns the current simulation time."""
        return float(self.data.time)

    def set_control(self, actuator_name: str, value: float) -> None:
        """Sets the control signal for a specific actuator."""
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        if actuator_id != -1:
            self.data.ctrl[actuator_id] = value
        else:
            logger.warning(f"Actuator {actuator_name} not found.")

    def get_joint_angle(self, joint_name: str) -> float:
        """Returns the position (angle) of a specific joint."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            # For 1-DOF joints, qpos is a single value
            qpos_addr = self.model.jnt_qposadr[joint_id]
            return float(self.data.qpos[qpos_addr])
        else:
            logger.warning(f"Joint {joint_name} not found.")
            return 0.0

    def get_joint_velocity(self, joint_name: str) -> float:
        """Returns the velocity of a specific joint."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            dof_addr = self.model.jnt_dofadr[joint_id]
            return float(self.data.qvel[dof_addr])
        else:
            logger.warning(f"Joint {joint_name} not found.")
            return 0.0
