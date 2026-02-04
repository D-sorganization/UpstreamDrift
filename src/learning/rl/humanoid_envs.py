"""Humanoid locomotion and balance environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.learning.rl.base_env import RoboticsGymEnv
from src.learning.rl.configs import (
    ActionConfig,
    ObservationConfig,
    RewardConfig,
    TaskConfig,
    TaskType,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


class HumanoidWalkEnv(RoboticsGymEnv):
    """Humanoid walking task environment.

    The agent controls a humanoid robot to walk at a target velocity.
    Rewards are based on forward progress, energy efficiency, and
    maintaining stable upright posture.

    Observation includes:
        - Joint positions and velocities
        - Base orientation and angular velocity
        - Contact states for feet

    Action is joint torques or position targets depending on config.
    """

    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        model_path: str | None = None,
        target_velocity: float = 1.0,
        obs_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize humanoid walking environment.

        Args:
            engine: Physics engine for simulation.
            model_path: Path to humanoid model.
            target_velocity: Target forward walking velocity (m/s).
            obs_config: Observation configuration.
            action_config: Action configuration.
            reward_config: Reward configuration.
            render_mode: Render mode.
        """
        # Create task config for walking
        task_config = TaskConfig(
            task_type=TaskType.LOCOMOTION,
            target_velocity=np.array([target_velocity, 0.0, 0.0]),
            max_episode_steps=1000,
            early_termination=True,
        )

        super().__init__(
            engine=engine,
            model_path=model_path,
            task_config=task_config,
            obs_config=obs_config,
            action_config=action_config,
            reward_config=reward_config,
            render_mode=render_mode,
        )

        # Walking specific state
        self._base_height_threshold = 0.3  # Minimum height before termination
        self._base_tilt_threshold = 0.8  # Max tilt (cos angle from vertical)

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        """Apply joint torques or positions to humanoid."""
        if hasattr(self.engine, "set_joint_torques"):
            self.engine.set_joint_torques(action)
        elif hasattr(self.engine, "set_control"):
            self.engine.set_control(action)

    def _step_simulation(self) -> None:
        """Advance simulation by one timestep."""
        if hasattr(self.engine, "step"):
            self.engine.step()

    def _get_observation(self) -> NDArray[np.floating]:
        """Get observation for walking task."""
        obs_parts = []

        # Joint positions
        if self.obs_config.include_joint_pos:
            if hasattr(self.engine, "get_joint_positions"):
                q = self.engine.get_joint_positions()
            else:
                q = np.zeros(self._n_joints)
            # Add noise
            if self.obs_config.position_noise_std > 0:
                q = q + self.np_random.normal(
                    0, self.obs_config.position_noise_std, q.shape
                )
            obs_parts.append(q)

        # Joint velocities
        if self.obs_config.include_joint_vel:
            if hasattr(self.engine, "get_joint_velocities"):
                qd = self.engine.get_joint_velocities()
            else:
                qd = np.zeros(self._n_joints)
            # Add noise
            if self.obs_config.velocity_noise_std > 0:
                qd = qd + self.np_random.normal(
                    0, self.obs_config.velocity_noise_std, qd.shape
                )
            obs_parts.append(qd)

        # Joint torques
        if self.obs_config.include_joint_torque:
            if hasattr(self.engine, "get_joint_torques"):
                tau = self.engine.get_joint_torques()
            else:
                tau = np.zeros(self._n_joints)
            obs_parts.append(tau)

        # IMU data (base orientation + angular velocity)
        if self.obs_config.include_imu:
            if hasattr(self.engine, "get_imu_data"):
                imu = self.engine.get_imu_data()
            else:
                imu = np.zeros(6)
            obs_parts.append(imu)

        # Contact forces
        if self.obs_config.include_contact_forces:
            if hasattr(self.engine, "get_contact_forces"):
                contacts = self.engine.get_contact_forces()
            else:
                contacts = np.zeros(6)  # Left and right foot
            obs_parts.append(contacts)

        obs = np.concatenate(obs_parts).astype(np.float32)
        return obs

    def _compute_reward(self, action: NDArray[np.floating]) -> float:
        """Compute reward for walking task."""
        reward = 0.0

        # Forward velocity reward
        if hasattr(self.engine, "get_base_velocity"):
            base_vel = self.engine.get_base_velocity()
        else:
            base_vel = np.zeros(3)

        target_vel = self.task_config.target_velocity
        vel_error = np.linalg.norm(base_vel[:2] - target_vel[:2])
        vel_reward = np.exp(-vel_error)
        reward += vel_reward * self.reward_config.task_reward_weight

        # Alive bonus
        reward += self.reward_config.alive_bonus

        # Energy penalty
        if hasattr(self.engine, "get_joint_torques"):
            torques = self.engine.get_joint_torques()
        else:
            torques = action
        reward -= self.reward_config.compute_energy_penalty(torques)

        # Smoothness penalty
        reward -= self.reward_config.compute_smoothness_penalty(
            action, self._prev_action
        )

        # Potential shaping
        if self.reward_config.use_potential_shaping:
            potential = self._compute_potential()
            reward += potential - self._prev_potential
            self._prev_potential = potential

        return float(reward)

    def _compute_potential(self) -> float:
        """Compute potential based on forward progress."""
        if hasattr(self.engine, "get_base_position"):
            pos = self.engine.get_base_position()
            return float(pos[0])  # Forward progress
        return 0.0

    def _check_termination(self) -> bool:
        """Check if humanoid has fallen."""
        if not self.task_config.early_termination:
            return False

        # Check base height
        if hasattr(self.engine, "get_base_position"):
            pos = self.engine.get_base_position()
            if pos[2] < self._base_height_threshold:
                return True

        # Check base tilt
        if hasattr(self.engine, "get_base_orientation"):
            quat = self.engine.get_base_orientation()
            # Check z-component of up vector after rotation
            # Simplified: check quaternion indicates large tilt
            up_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
            if up_z < self._base_tilt_threshold:
                return True

        return False

    def _reset_simulation(self, options: dict[str, Any] | None) -> None:
        """Reset humanoid to standing position."""
        if hasattr(self.engine, "reset"):
            self.engine.reset()

        # Apply small random perturbation to initial state
        if options and options.get("randomize_init", True):
            noise_scale = options.get("init_noise_scale", 0.01)
            if hasattr(self.engine, "get_joint_positions"):
                q = self.engine.get_joint_positions()
                q_noise = self.np_random.normal(0, noise_scale, q.shape)
                if hasattr(self.engine, "set_joint_positions"):
                    self.engine.set_joint_positions(q + q_noise)

    def _get_info(self) -> dict[str, Any]:
        """Get walking-specific info."""
        info = super()._get_info()

        if hasattr(self.engine, "get_base_velocity"):
            base_vel = self.engine.get_base_velocity()
            info["forward_velocity"] = float(base_vel[0])
            info["lateral_velocity"] = float(base_vel[1])

        if hasattr(self.engine, "get_base_position"):
            pos = self.engine.get_base_position()
            info["base_height"] = float(pos[2])
            info["forward_distance"] = float(pos[0])

        return info


class HumanoidStandEnv(RoboticsGymEnv):
    """Humanoid standing/balance task environment.

    The agent must maintain balance while standing in place,
    potentially while experiencing external perturbations.

    Rewards maintaining upright posture with minimal movement
    and low energy consumption.
    """

    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        model_path: str | None = None,
        perturbation_force: float = 0.0,
        obs_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize humanoid standing environment.

        Args:
            engine: Physics engine for simulation.
            model_path: Path to humanoid model.
            perturbation_force: Max external perturbation force (N).
            obs_config: Observation configuration.
            action_config: Action configuration.
            reward_config: Reward configuration.
            render_mode: Render mode.
        """
        task_config = TaskConfig(
            task_type=TaskType.BALANCE,
            target_velocity=np.array([0.0, 0.0, 0.0]),
            max_episode_steps=500,
            early_termination=True,
        )

        super().__init__(
            engine=engine,
            model_path=model_path,
            task_config=task_config,
            obs_config=obs_config,
            action_config=action_config,
            reward_config=reward_config,
            render_mode=render_mode,
        )

        self._perturbation_force = perturbation_force
        self._base_height_threshold = 0.3
        self._target_height = 0.9  # Standing height

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        """Apply joint control to humanoid."""
        if hasattr(self.engine, "set_joint_torques"):
            self.engine.set_joint_torques(action)

    def _step_simulation(self) -> None:
        """Step simulation, possibly applying perturbation."""
        # Random perturbation
        if self._perturbation_force > 0:
            if self.np_random.random() < 0.01:  # 1% chance per step
                force = self.np_random.uniform(
                    -self._perturbation_force,
                    self._perturbation_force,
                    size=3,
                )
                force[2] = 0  # No vertical force
                if hasattr(self.engine, "apply_external_force"):
                    self.engine.apply_external_force("torso", force)

        if hasattr(self.engine, "step"):
            self.engine.step()

    def _get_observation(self) -> NDArray[np.floating]:
        """Get observation for standing task."""
        obs_parts = []

        if self.obs_config.include_joint_pos:
            if hasattr(self.engine, "get_joint_positions"):
                q = self.engine.get_joint_positions()
            else:
                q = np.zeros(self._n_joints)
            obs_parts.append(q)

        if self.obs_config.include_joint_vel:
            if hasattr(self.engine, "get_joint_velocities"):
                qd = self.engine.get_joint_velocities()
            else:
                qd = np.zeros(self._n_joints)
            obs_parts.append(qd)

        if self.obs_config.include_imu:
            if hasattr(self.engine, "get_imu_data"):
                imu = self.engine.get_imu_data()
            else:
                imu = np.zeros(6)
            obs_parts.append(imu)

        return np.concatenate(obs_parts).astype(np.float32)

    def _compute_reward(self, action: NDArray[np.floating]) -> float:
        """Compute reward for standing task."""
        reward = 0.0

        # Height maintenance reward
        if hasattr(self.engine, "get_base_position"):
            pos = self.engine.get_base_position()
            height_error = abs(pos[2] - self._target_height)
            height_reward = np.exp(-10 * height_error)
            reward += height_reward * self.reward_config.task_reward_weight

        # Upright orientation reward
        if hasattr(self.engine, "get_base_orientation"):
            quat = self.engine.get_base_orientation()
            up_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
            upright_reward = (up_z + 1) / 2  # Normalize to [0, 1]
            reward += upright_reward * 0.5

        # Minimal movement reward (stay still)
        if hasattr(self.engine, "get_base_velocity"):
            vel = self.engine.get_base_velocity()
            stillness_reward = np.exp(-np.linalg.norm(vel))
            reward += stillness_reward * 0.3

        # Alive bonus
        reward += self.reward_config.alive_bonus

        # Energy penalty
        if hasattr(self.engine, "get_joint_torques"):
            torques = self.engine.get_joint_torques()
        else:
            torques = action
        reward -= self.reward_config.compute_energy_penalty(torques)

        return float(reward)

    def _check_termination(self) -> bool:
        """Check if humanoid has fallen."""
        if not self.task_config.early_termination:
            return False

        if hasattr(self.engine, "get_base_position"):
            pos = self.engine.get_base_position()
            if pos[2] < self._base_height_threshold:
                return True

        return False

    def _reset_simulation(self, options: dict[str, Any] | None) -> None:
        """Reset humanoid to standing position."""
        if hasattr(self.engine, "reset"):
            self.engine.reset()

    def _get_info(self) -> dict[str, Any]:
        """Get standing-specific info."""
        info = super()._get_info()

        if hasattr(self.engine, "get_base_position"):
            pos = self.engine.get_base_position()
            info["base_height"] = float(pos[2])

        if hasattr(self.engine, "get_base_orientation"):
            quat = self.engine.get_base_orientation()
            up_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
            info["upright_score"] = float(up_z)

        return info
