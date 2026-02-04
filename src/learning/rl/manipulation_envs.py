"""Manipulation task environments for robotic arms."""

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


class ManipulationPickPlaceEnv(RoboticsGymEnv):
    """Pick and place manipulation environment.

    The robot arm must pick up an object from one location
    and place it at a target location.

    Observation includes:
        - Joint positions and velocities
        - End-effector position and orientation
        - Object position
        - Target position

    Action is joint torques or positions depending on config.
    """

    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        model_path: str | None = None,
        object_initial_pos: NDArray[np.floating] | None = None,
        target_pos: NDArray[np.floating] | None = None,
        obs_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize pick and place environment.

        Args:
            engine: Physics engine for simulation.
            model_path: Path to robot model.
            object_initial_pos: Initial position of object to pick.
            target_pos: Target position for placing.
            obs_config: Observation configuration.
            action_config: Action configuration.
            reward_config: Reward configuration.
            render_mode: Render mode.
        """
        self._object_pos = object_initial_pos or np.array([0.5, 0.0, 0.1])
        self._target_pos = target_pos or np.array([0.5, 0.3, 0.1])

        task_config = TaskConfig(
            task_type=TaskType.MANIPULATION,
            target_position=self._target_pos,
            max_episode_steps=500,
            early_termination=False,
            success_threshold=0.05,
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

        # Task state
        self._object_grasped = False
        self._grasp_threshold = 0.05  # Distance for successful grasp
        self._place_threshold = 0.05  # Distance for successful place

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        """Apply action to robot arm."""
        # Split action into arm control and gripper
        arm_action = action[:-1] if len(action) > self._n_actuators - 1 else action
        gripper_action = action[-1] if len(action) > self._n_actuators - 1 else 0.0

        if hasattr(self.engine, "set_joint_torques"):
            self.engine.set_joint_torques(arm_action)

        if hasattr(self.engine, "set_gripper"):
            self.engine.set_gripper(gripper_action)

    def _step_simulation(self) -> None:
        """Step simulation and update grasp state."""
        if hasattr(self.engine, "step"):
            self.engine.step()

        # Update grasp state
        self._update_grasp_state()

    def _update_grasp_state(self) -> None:
        """Update whether object is grasped."""
        if hasattr(self.engine, "get_ee_position"):
            ee_pos = self.engine.get_ee_position()
            obj_pos = self._get_object_position()
            dist = np.linalg.norm(ee_pos - obj_pos)

            # Check if gripper is closed and near object
            gripper_closed = self._is_gripper_closed()
            if dist < self._grasp_threshold and gripper_closed:
                self._object_grasped = True
            elif not gripper_closed:
                self._object_grasped = False

    def _is_gripper_closed(self) -> bool:
        """Check if gripper is in closed state."""
        if hasattr(self.engine, "get_gripper_state"):
            return self.engine.get_gripper_state() < 0.1
        return False

    def _get_object_position(self) -> NDArray[np.floating]:
        """Get current object position."""
        if self._object_grasped and hasattr(self.engine, "get_ee_position"):
            # Object follows end-effector when grasped
            return self.engine.get_ee_position()
        if hasattr(self.engine, "get_object_position"):
            return self.engine.get_object_position("target_object")
        return self._object_pos.copy()

    def _get_observation(self) -> NDArray[np.floating]:
        """Get observation for manipulation task."""
        obs_parts = []

        # Joint state
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

        # End-effector state
        if self.obs_config.include_ee_pos:
            if hasattr(self.engine, "get_ee_position"):
                ee_pos = self.engine.get_ee_position()
            else:
                ee_pos = np.zeros(3)
            obs_parts.append(ee_pos)

        if self.obs_config.include_ee_vel:
            if hasattr(self.engine, "get_ee_velocity"):
                ee_vel = self.engine.get_ee_velocity()
            else:
                ee_vel = np.zeros(6)
            obs_parts.append(ee_vel)

        # Object and target positions
        obj_pos = self._get_object_position()
        obs_parts.append(obj_pos)
        obs_parts.append(self._target_pos)

        # Grasp state
        obs_parts.append(np.array([float(self._object_grasped)]))

        return np.concatenate(obs_parts).astype(np.float32)

    def _build_observation_space(self):
        """Build observation space including object state."""
        from gymnasium import spaces

        base_dim = self.obs_config.get_obs_dim(self._n_joints, self._n_ee)
        # Add object pos (3), target pos (3), grasp state (1)
        total_dim = base_dim + 7
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def _compute_reward(self, action: NDArray[np.floating]) -> float:
        """Compute reward for pick and place task."""
        reward = 0.0

        obj_pos = self._get_object_position()
        target_pos = self._target_pos

        if hasattr(self.engine, "get_ee_position"):
            ee_pos = self.engine.get_ee_position()
        else:
            ee_pos = np.zeros(3)

        if not self._object_grasped:
            # Reaching reward: get close to object
            reach_dist = np.linalg.norm(ee_pos - obj_pos)
            reach_reward = np.exp(-5 * reach_dist)
            reward += reach_reward * self.reward_config.task_reward_weight * 0.5
        else:
            # Placing reward: move object to target
            place_dist = np.linalg.norm(obj_pos - target_pos)
            place_reward = np.exp(-5 * place_dist)
            reward += place_reward * self.reward_config.task_reward_weight

            # Bonus for successful placement
            if place_dist < self._place_threshold:
                reward += 10.0

        # Energy penalty
        reward -= self.reward_config.compute_energy_penalty(action)

        # Smoothness penalty
        reward -= self.reward_config.compute_smoothness_penalty(
            action, self._prev_action
        )

        return float(reward)

    def _check_termination(self) -> bool:
        """Check if task is complete."""
        if not self._object_grasped:
            return False

        obj_pos = self._get_object_position()
        dist_to_target = np.linalg.norm(obj_pos - self._target_pos)

        return dist_to_target < self._place_threshold

    def _reset_simulation(self, options: dict[str, Any] | None) -> None:
        """Reset manipulation environment."""
        if hasattr(self.engine, "reset"):
            self.engine.reset()

        self._object_grasped = False

        # Optionally randomize positions
        if options and options.get("randomize_positions", False):
            self._object_pos = self._object_pos + self.np_random.uniform(
                -0.1, 0.1, size=3
            )
            self._target_pos = self._target_pos + self.np_random.uniform(
                -0.1, 0.1, size=3
            )

    def _get_info(self) -> dict[str, Any]:
        """Get manipulation-specific info."""
        info = super()._get_info()

        obj_pos = self._get_object_position()
        info["object_grasped"] = self._object_grasped
        info["object_position"] = obj_pos.tolist()
        info["target_position"] = self._target_pos.tolist()
        info["distance_to_target"] = float(np.linalg.norm(obj_pos - self._target_pos))

        return info


class DualArmManipulationEnv(RoboticsGymEnv):
    """Bimanual manipulation environment with two robot arms.

    The agent controls two arms to perform coordinated manipulation
    tasks such as lifting large objects or assembly operations.

    Observation includes both arms' states plus shared object state.
    Action space is concatenation of both arms' actions.
    """

    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        model_path: str | None = None,
        object_mass: float = 5.0,
        obs_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize dual arm manipulation environment.

        Args:
            engine: Physics engine for simulation.
            model_path: Path to dual-arm robot model.
            object_mass: Mass of object to manipulate (kg).
            obs_config: Observation configuration.
            action_config: Action configuration.
            reward_config: Reward configuration.
            render_mode: Render mode.
        """
        task_config = TaskConfig(
            task_type=TaskType.MANIPULATION,
            target_position=np.array([0.5, 0.0, 0.5]),
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

        self._object_mass = object_mass
        self._left_grasped = False
        self._right_grasped = False
        self._object_lifted = False

    def _get_n_end_effectors(self) -> int:
        """Dual arm has two end-effectors."""
        return 2

    def _build_action_space(self):
        """Build action space for both arms."""
        from gymnasium import spaces

        # Double the action space for two arms
        clip = self.action_config.action_clip
        return spaces.Box(
            low=-clip,
            high=clip,
            shape=(self._n_actuators * 2,),
            dtype=np.float32,
        )

    def _apply_action(self, action: NDArray[np.floating]) -> None:
        """Apply actions to both arms."""
        n = len(action) // 2
        left_action = action[:n]
        right_action = action[n:]

        if hasattr(self.engine, "set_left_arm_torques"):
            self.engine.set_left_arm_torques(left_action)
        if hasattr(self.engine, "set_right_arm_torques"):
            self.engine.set_right_arm_torques(right_action)

    def _step_simulation(self) -> None:
        """Step simulation and update grasp states."""
        if hasattr(self.engine, "step"):
            self.engine.step()

        # Update grasp states
        self._update_dual_grasp_state()

    def _update_dual_grasp_state(self) -> None:
        """Update grasp state for both arms."""
        grasp_threshold = 0.08

        if hasattr(self.engine, "get_left_ee_position"):
            left_ee = self.engine.get_left_ee_position()
            obj_pos = self._get_object_position()
            left_dist = np.linalg.norm(left_ee - obj_pos)
            left_closed = self._is_left_gripper_closed()
            self._left_grasped = left_dist < grasp_threshold and left_closed

        if hasattr(self.engine, "get_right_ee_position"):
            right_ee = self.engine.get_right_ee_position()
            obj_pos = self._get_object_position()
            right_dist = np.linalg.norm(right_ee - obj_pos)
            right_closed = self._is_right_gripper_closed()
            self._right_grasped = right_dist < grasp_threshold and right_closed

        # Object is lifted if both arms are grasping
        self._object_lifted = self._left_grasped and self._right_grasped

    def _is_left_gripper_closed(self) -> bool:
        """Check if left gripper is closed."""
        if hasattr(self.engine, "get_left_gripper_state"):
            return self.engine.get_left_gripper_state() < 0.1
        return False

    def _is_right_gripper_closed(self) -> bool:
        """Check if right gripper is closed."""
        if hasattr(self.engine, "get_right_gripper_state"):
            return self.engine.get_right_gripper_state() < 0.1
        return False

    def _get_object_position(self) -> NDArray[np.floating]:
        """Get current object position."""
        if hasattr(self.engine, "get_object_position"):
            return self.engine.get_object_position("heavy_object")
        return np.array([0.5, 0.0, 0.1])

    def _get_observation(self) -> NDArray[np.floating]:
        """Get observation for dual arm task."""
        obs_parts = []

        # Left arm state
        if self.obs_config.include_joint_pos:
            if hasattr(self.engine, "get_left_joint_positions"):
                q_left = self.engine.get_left_joint_positions()
            else:
                q_left = np.zeros(self._n_joints // 2)
            obs_parts.append(q_left)

        if self.obs_config.include_joint_vel:
            if hasattr(self.engine, "get_left_joint_velocities"):
                qd_left = self.engine.get_left_joint_velocities()
            else:
                qd_left = np.zeros(self._n_joints // 2)
            obs_parts.append(qd_left)

        # Right arm state
        if self.obs_config.include_joint_pos:
            if hasattr(self.engine, "get_right_joint_positions"):
                q_right = self.engine.get_right_joint_positions()
            else:
                q_right = np.zeros(self._n_joints // 2)
            obs_parts.append(q_right)

        if self.obs_config.include_joint_vel:
            if hasattr(self.engine, "get_right_joint_velocities"):
                qd_right = self.engine.get_right_joint_velocities()
            else:
                qd_right = np.zeros(self._n_joints // 2)
            obs_parts.append(qd_right)

        # Object state
        obj_pos = self._get_object_position()
        obs_parts.append(obj_pos)
        obs_parts.append(self.task_config.target_position)

        # Grasp states
        grasp_state = np.array(
            [
                float(self._left_grasped),
                float(self._right_grasped),
                float(self._object_lifted),
            ]
        )
        obs_parts.append(grasp_state)

        return np.concatenate(obs_parts).astype(np.float32)

    def _compute_reward(self, action: NDArray[np.floating]) -> float:
        """Compute reward for coordinated manipulation."""
        reward = 0.0

        obj_pos = self._get_object_position()
        target_pos = self.task_config.target_position

        # Coordination reward: both arms approaching object
        if not self._object_lifted:
            # Get EE positions
            if hasattr(self.engine, "get_left_ee_position"):
                left_ee = self.engine.get_left_ee_position()
            else:
                left_ee = np.zeros(3)

            if hasattr(self.engine, "get_right_ee_position"):
                right_ee = self.engine.get_right_ee_position()
            else:
                right_ee = np.zeros(3)

            left_dist = np.linalg.norm(left_ee - obj_pos)
            right_dist = np.linalg.norm(right_ee - obj_pos)

            reach_reward = np.exp(-3 * (left_dist + right_dist))
            reward += reach_reward * self.reward_config.task_reward_weight * 0.3

            # Bonus for successful grasp
            if self._left_grasped:
                reward += 1.0
            if self._right_grasped:
                reward += 1.0
        else:
            # Lifting/placing reward
            dist_to_target = np.linalg.norm(obj_pos - target_pos)
            lift_reward = np.exp(-3 * dist_to_target)
            reward += lift_reward * self.reward_config.task_reward_weight

            # Large bonus for reaching target
            if dist_to_target < 0.1:
                reward += 20.0

        # Coordination penalty: arms should move together
        if hasattr(self.engine, "get_left_ee_velocity") and hasattr(
            self.engine, "get_right_ee_velocity"
        ):
            left_vel = self.engine.get_left_ee_velocity()[:3]
            right_vel = self.engine.get_right_ee_velocity()[:3]
            vel_diff = np.linalg.norm(left_vel - right_vel)
            reward -= 0.1 * vel_diff

        # Energy penalty
        reward -= self.reward_config.compute_energy_penalty(action)

        return float(reward)

    def _check_termination(self) -> bool:
        """Check for task failure (dropped object)."""
        if self._object_lifted:
            obj_pos = self._get_object_position()
            # Terminate if object falls
            if obj_pos[2] < 0.05:
                return True
        return False

    def _reset_simulation(self, options: dict[str, Any] | None) -> None:
        """Reset dual arm environment."""
        if hasattr(self.engine, "reset"):
            self.engine.reset()

        self._left_grasped = False
        self._right_grasped = False
        self._object_lifted = False

    def _get_info(self) -> dict[str, Any]:
        """Get dual arm specific info."""
        info = super()._get_info()

        obj_pos = self._get_object_position()
        info["left_grasped"] = self._left_grasped
        info["right_grasped"] = self._right_grasped
        info["object_lifted"] = self._object_lifted
        info["object_position"] = obj_pos.tolist()
        info["distance_to_target"] = float(
            np.linalg.norm(obj_pos - self.task_config.target_position)
        )

        return info
