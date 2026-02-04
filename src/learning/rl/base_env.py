"""Base Gymnasium environment for robotics simulation."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None

from src.learning.rl.configs import (
    ActionConfig,
    ActionMode,
    ObservationConfig,
    RewardConfig,
    TaskConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


class RoboticsGymEnv:
    """Gymnasium environment wrapping UpstreamDrift simulation.

    This base class provides the core functionality for creating
    RL environments using the physics engines in UpstreamDrift.
    Subclasses implement specific tasks like walking or manipulation.

    Attributes:
        engine: The physics engine instance.
        task_config: Task configuration.
        obs_config: Observation configuration.
        action_config: Action configuration.
        reward_config: Reward configuration.
        observation_space: Gymnasium observation space.
        action_space: Gymnasium action space.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        model_path: str | None = None,
        task_config: TaskConfig | None = None,
        obs_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the robotics environment.

        Args:
            engine: Physics engine to use for simulation.
            model_path: Path to robot model file.
            task_config: Task configuration.
            obs_config: Observation configuration.
            action_config: Action configuration.
            reward_config: Reward configuration.
            render_mode: Render mode ("human" or "rgb_array").
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for RL environments. "
                "Install with: pip install gymnasium"
            )

        self.engine = engine
        self.model_path = model_path
        self.render_mode = render_mode

        # Load model if path provided
        if model_path and hasattr(engine, "load_from_path"):
            engine.load_from_path(model_path)

        # Use default configs if not provided
        self.task_config = task_config or TaskConfig()
        self.obs_config = obs_config or ObservationConfig()
        self.action_config = action_config or ActionConfig()
        self.reward_config = reward_config or RewardConfig()

        # Get dimensions from engine
        self._n_joints = self._get_n_joints()
        self._n_actuators = self._get_n_actuators()
        self._n_ee = self._get_n_end_effectors()

        # Build spaces
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

        # Episode state
        self._step_count = 0
        self._prev_action: NDArray[np.floating] | None = None
        self._prev_potential: float = 0.0

        # Initialize random generator
        self._np_random: np.random.Generator | None = None

    def _get_n_joints(self) -> int:
        """Get number of joints from engine."""
        if hasattr(self.engine, "n_q"):
            return int(self.engine.n_q)
        return 7  # Default for 7-DOF arm

    def _get_n_actuators(self) -> int:
        """Get number of actuators from engine."""
        if hasattr(self.engine, "n_v"):
            return int(self.engine.n_v)
        return self._n_joints

    def _get_n_end_effectors(self) -> int:
        """Get number of end-effectors."""
        return 1  # Override in subclasses

    def _build_observation_space(self) -> spaces.Box:
        """Build the observation space based on config.

        Returns:
            Gymnasium Box space for observations.
        """
        obs_dim = self.obs_config.get_obs_dim(self._n_joints, self._n_ee)
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _build_action_space(self) -> spaces.Box:
        """Build the action space based on config.

        Returns:
            Gymnasium Box space for actions.
        """
        clip = self.action_config.action_clip
        return spaces.Box(
            low=-clip,
            high=clip,
            shape=(self._n_actuators,),
            dtype=np.float32,
        )

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action from the policy.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Process action
        processed_action = self.action_config.process_action(
            action, self._prev_action
        )

        # Apply action to simulation
        self._apply_action(processed_action)

        # Step simulation
        self._step_simulation()

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(processed_action)

        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= self.task_config.max_episode_steps

        # Build info dict
        info = self._get_info()

        # Update state
        self._step_count += 1
        self._prev_action = processed_action.copy()

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (initial observation, info dict).
        """
        # Handle seeding
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        # Reset simulation state
        self._reset_simulation(options)

        # Reset episode state
        self._step_count = 0
        self._prev_action = None
        self._prev_potential = self._compute_potential()

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_human()
            return None
        return None

    def close(self) -> None:
        """Clean up environment resources."""
        if hasattr(self.engine, "close"):
            self.engine.close()

    @property
    def np_random(self) -> np.random.Generator:
        """Get the random number generator."""
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    def _apply_action(self, action: NDArray[np.floating]) -> None:
        """Apply action to simulation.

        Args:
            action: Processed action to apply.
        """
        pass

    @abstractmethod
    def _step_simulation(self) -> None:
        """Advance the simulation by one timestep."""
        pass

    @abstractmethod
    def _get_observation(self) -> NDArray[np.floating]:
        """Get current observation from simulation.

        Returns:
            Observation array.
        """
        pass

    @abstractmethod
    def _compute_reward(self, action: NDArray[np.floating]) -> float:
        """Compute reward for current state.

        Args:
            action: Applied action.

        Returns:
            Reward value.
        """
        pass

    @abstractmethod
    def _check_termination(self) -> bool:
        """Check if episode should terminate.

        Returns:
            True if episode should terminate.
        """
        pass

    @abstractmethod
    def _reset_simulation(self, options: dict[str, Any] | None) -> None:
        """Reset simulation to initial state.

        Args:
            options: Reset options.
        """
        pass

    def _compute_potential(self) -> float:
        """Compute potential function for reward shaping.

        Returns:
            Potential value (override in subclasses).
        """
        return 0.0

    def _get_info(self) -> dict[str, Any]:
        """Get additional info dict.

        Returns:
            Info dictionary.
        """
        return {
            "step_count": self._step_count,
        }

    def _render_frame(self) -> NDArray[np.uint8]:
        """Render a frame as RGB array.

        Returns:
            RGB image array.
        """
        if hasattr(self.engine, "render"):
            return self.engine.render()
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _render_human(self) -> None:
        """Render to human display."""
        if hasattr(self.engine, "render_to_screen"):
            self.engine.render_to_screen()
