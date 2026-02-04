"""Configuration dataclasses for RL environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ActionMode(Enum):
    """Action space mode for the environment."""

    TORQUE = "torque"
    POSITION = "position"
    VELOCITY = "velocity"
    IMPEDANCE = "impedance"


class TaskType(Enum):
    """Type of task for the environment."""

    LOCOMOTION = "locomotion"
    MANIPULATION = "manipulation"
    BALANCE = "balance"
    TRACKING = "tracking"
    REACHING = "reaching"


@dataclass
class ObservationConfig:
    """Configure observation space for RL environments.

    Attributes:
        include_joint_pos: Include joint positions in observation.
        include_joint_vel: Include joint velocities in observation.
        include_joint_torque: Include joint torques in observation.
        include_ee_pos: Include end-effector positions.
        include_ee_vel: Include end-effector velocities.
        include_contact_forces: Include contact force readings.
        include_imu: Include IMU sensor data.
        include_privileged: Include privileged info (for teacher policies).
        position_noise_std: Standard deviation of position noise.
        velocity_noise_std: Standard deviation of velocity noise.
        history_length: Number of previous observations to stack.
    """

    include_joint_pos: bool = True
    include_joint_vel: bool = True
    include_joint_torque: bool = False
    include_ee_pos: bool = False
    include_ee_vel: bool = False
    include_contact_forces: bool = False
    include_imu: bool = False
    include_privileged: bool = False

    # Noise models
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0

    # History
    history_length: int = 1

    def get_obs_dim(self, n_joints: int, n_ee: int = 0) -> int:
        """Calculate observation dimension based on config.

        Args:
            n_joints: Number of joints in the robot.
            n_ee: Number of end-effectors.

        Returns:
            Total observation dimension.
        """
        dim = 0
        if self.include_joint_pos:
            dim += n_joints
        if self.include_joint_vel:
            dim += n_joints
        if self.include_joint_torque:
            dim += n_joints
        if self.include_ee_pos:
            dim += n_ee * 3
        if self.include_ee_vel:
            dim += n_ee * 6
        if self.include_imu:
            dim += 6  # 3 accel + 3 gyro
        return dim * self.history_length


@dataclass
class ActionConfig:
    """Configure action space for RL environments.

    Attributes:
        mode: Action mode (torque, position, velocity, impedance).
        action_scale: Scale factor for actions.
        action_clip: Maximum absolute action value.
        smoothing_alpha: Exponential smoothing factor (0=no smoothing).
    """

    mode: ActionMode = ActionMode.TORQUE
    action_scale: float = 1.0
    action_clip: float = 1.0
    smoothing_alpha: float = 0.0

    def process_action(
        self, action: NDArray[np.floating], prev_action: NDArray[np.floating] | None
    ) -> NDArray[np.floating]:
        """Process raw action through scaling and smoothing.

        Args:
            action: Raw action from policy.
            prev_action: Previous action for smoothing.

        Returns:
            Processed action.
        """
        # Clip
        action = np.clip(action, -self.action_clip, self.action_clip)
        # Scale
        action = action * self.action_scale
        # Smooth
        if self.smoothing_alpha > 0 and prev_action is not None:
            action = (
                self.smoothing_alpha * prev_action
                + (1 - self.smoothing_alpha) * action
            )
        return action


@dataclass
class RewardConfig:
    """Configure reward function for RL environments.

    Attributes:
        task_reward_weight: Weight for task-specific reward.
        energy_penalty_weight: Penalty weight for energy consumption.
        smoothness_penalty_weight: Penalty for jerky motions.
        contact_penalty_weight: Penalty for undesired contacts.
        use_potential_shaping: Use potential-based reward shaping.
        alive_bonus: Constant bonus for not terminating.
    """

    task_reward_weight: float = 1.0
    energy_penalty_weight: float = 0.001
    smoothness_penalty_weight: float = 0.0001
    contact_penalty_weight: float = 0.0

    # Shaping
    use_potential_shaping: bool = False
    alive_bonus: float = 0.0

    def compute_energy_penalty(self, torques: NDArray[np.floating]) -> float:
        """Compute energy penalty from applied torques.

        Args:
            torques: Applied joint torques.

        Returns:
            Energy penalty value.
        """
        return float(np.sum(torques**2)) * self.energy_penalty_weight

    def compute_smoothness_penalty(
        self,
        action: NDArray[np.floating],
        prev_action: NDArray[np.floating] | None,
    ) -> float:
        """Compute smoothness penalty from action changes.

        Args:
            action: Current action.
            prev_action: Previous action.

        Returns:
            Smoothness penalty value.
        """
        if prev_action is None:
            return 0.0
        diff = action - prev_action
        return float(np.sum(diff**2)) * self.smoothness_penalty_weight


@dataclass
class TaskConfig:
    """Configure task parameters for RL environments.

    Attributes:
        task_type: Type of task.
        target_velocity: Target velocity for locomotion tasks.
        target_position: Target position for manipulation tasks.
        target_orientation: Target orientation (quaternion).
        max_episode_steps: Maximum steps per episode.
        early_termination: Whether to terminate on failure.
        success_threshold: Threshold for considering task successful.
    """

    task_type: TaskType = TaskType.LOCOMOTION
    target_velocity: NDArray[np.floating] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    target_position: NDArray[np.floating] | None = None
    target_orientation: NDArray[np.floating] | None = None
    max_episode_steps: int = 1000
    early_termination: bool = True
    success_threshold: float = 0.05

    def is_success(self, error: float) -> bool:
        """Check if task is successfully completed.

        Args:
            error: Current task error.

        Returns:
            True if error is below success threshold.
        """
        return error < self.success_threshold
