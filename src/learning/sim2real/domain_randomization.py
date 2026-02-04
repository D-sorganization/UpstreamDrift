"""Domain Randomization for sim-to-real transfer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization.

    Randomization ranges are specified as (min_multiplier, max_multiplier)
    relative to nominal values.

    Attributes:
        mass_range: Range for link mass randomization.
        friction_range: Range for friction coefficient.
        damping_range: Range for joint damping.
        motor_strength_range: Range for motor strength.
        action_delay_range: Range for action delay (timesteps).
        observation_delay_range: Range for observation delay.
        observation_noise_std: Std dev of observation noise.
        action_noise_std: Std dev of action noise.
        gravity_range: Range for gravity magnitude.
        floor_friction_range: Range for floor friction.
        inertia_range: Range for link inertia.
        center_of_mass_range: Range for CoM position offset.
        joint_offset_range: Range for joint position offsets.
        actuator_gain_range: Range for actuator gains.
    """

    # Dynamics parameters
    mass_range: tuple[float, float] = (0.8, 1.2)
    friction_range: tuple[float, float] = (0.5, 1.5)
    damping_range: tuple[float, float] = (0.8, 1.2)
    motor_strength_range: tuple[float, float] = (0.9, 1.1)
    inertia_range: tuple[float, float] = (0.9, 1.1)
    center_of_mass_range: tuple[float, float] = (-0.01, 0.01)

    # Delays and noise
    action_delay_range: tuple[int, int] = (0, 3)
    observation_delay_range: tuple[int, int] = (0, 2)
    observation_noise_std: float = 0.01
    action_noise_std: float = 0.01

    # Environment
    gravity_range: tuple[float, float] = (9.5, 10.1)
    floor_friction_range: tuple[float, float] = (0.5, 1.5)

    # Actuator
    joint_offset_range: tuple[float, float] = (-0.02, 0.02)
    actuator_gain_range: tuple[float, float] = (0.95, 1.05)

    # Which parameters to randomize
    randomize_mass: bool = True
    randomize_friction: bool = True
    randomize_damping: bool = True
    randomize_motor: bool = True
    randomize_gravity: bool = True
    randomize_delays: bool = True
    randomize_noise: bool = True


class DomainRandomizer:
    """Apply domain randomization to physics simulation.

    Domain randomization varies physical parameters during training
    to improve robustness of learned policies when transferring
    to real robots.

    Attributes:
        engine: Physics engine to randomize.
        config: Randomization configuration.
        nominal_params: Stored nominal parameter values.
    """

    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        config: DomainRandomizationConfig | None = None,
    ) -> None:
        """Initialize domain randomizer.

        Args:
            engine: Physics engine instance.
            config: Randomization configuration.
        """
        self.engine = engine
        self.config = config or DomainRandomizationConfig()
        self.nominal_params = self._store_nominal_params()
        self._rng = np.random.default_rng()
        self._current_randomization: dict[str, Any] = {}

        # Delay buffers
        self._action_buffer: list[NDArray[np.floating]] = []
        self._observation_buffer: list[NDArray[np.floating]] = []
        self._action_delay = 0
        self._observation_delay = 0

    def _store_nominal_params(self) -> dict[str, Any]:
        """Store nominal parameter values from engine.

        Returns:
            Dictionary of nominal parameters.
        """
        params = {}

        # Get link masses if available
        if hasattr(self.engine, "get_link_masses"):
            params["masses"] = self.engine.get_link_masses().copy()

        # Get joint damping
        if hasattr(self.engine, "get_joint_damping"):
            params["damping"] = self.engine.get_joint_damping().copy()

        # Get friction coefficients
        if hasattr(self.engine, "get_friction_coefficients"):
            params["friction"] = self.engine.get_friction_coefficients().copy()

        # Get motor parameters
        if hasattr(self.engine, "get_motor_strength"):
            params["motor_strength"] = self.engine.get_motor_strength().copy()

        # Get gravity
        if hasattr(self.engine, "get_gravity"):
            params["gravity"] = self.engine.get_gravity().copy()

        # Get actuator gains
        if hasattr(self.engine, "get_actuator_gains"):
            params["actuator_gains"] = self.engine.get_actuator_gains().copy()

        return params

    def randomize(self, seed: int | None = None) -> dict[str, Any]:
        """Randomize simulation parameters.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Dictionary of applied randomization values.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        randomization = {}

        # Randomize mass
        if self.config.randomize_mass and "masses" in self.nominal_params:
            scale = self._rng.uniform(*self.config.mass_range)
            new_masses = self.nominal_params["masses"] * scale
            if hasattr(self.engine, "set_link_masses"):
                self.engine.set_link_masses(new_masses)
            randomization["mass_scale"] = scale

        # Randomize friction
        if self.config.randomize_friction and "friction" in self.nominal_params:
            scale = self._rng.uniform(*self.config.friction_range)
            new_friction = self.nominal_params["friction"] * scale
            if hasattr(self.engine, "set_friction_coefficients"):
                self.engine.set_friction_coefficients(new_friction)
            randomization["friction_scale"] = scale

        # Randomize damping
        if self.config.randomize_damping and "damping" in self.nominal_params:
            scale = self._rng.uniform(*self.config.damping_range)
            new_damping = self.nominal_params["damping"] * scale
            if hasattr(self.engine, "set_joint_damping"):
                self.engine.set_joint_damping(new_damping)
            randomization["damping_scale"] = scale

        # Randomize motor strength
        if self.config.randomize_motor and "motor_strength" in self.nominal_params:
            scale = self._rng.uniform(*self.config.motor_strength_range)
            new_strength = self.nominal_params["motor_strength"] * scale
            if hasattr(self.engine, "set_motor_strength"):
                self.engine.set_motor_strength(new_strength)
            randomization["motor_scale"] = scale

        # Randomize gravity
        if self.config.randomize_gravity and "gravity" in self.nominal_params:
            g_mag = self._rng.uniform(*self.config.gravity_range)
            gravity = self.nominal_params["gravity"].copy()
            gravity[2] = -g_mag  # Assuming z is up
            if hasattr(self.engine, "set_gravity"):
                self.engine.set_gravity(gravity)
            randomization["gravity"] = g_mag

        # Randomize delays
        if self.config.randomize_delays:
            self._action_delay = self._rng.integers(
                self.config.action_delay_range[0],
                self.config.action_delay_range[1] + 1,
            )
            self._observation_delay = self._rng.integers(
                self.config.observation_delay_range[0],
                self.config.observation_delay_range[1] + 1,
            )
            randomization["action_delay"] = self._action_delay
            randomization["observation_delay"] = self._observation_delay

            # Reset delay buffers
            self._action_buffer = []
            self._observation_buffer = []

        self._current_randomization = randomization
        return randomization

    def reset_to_nominal(self) -> None:
        """Reset all parameters to nominal values."""
        if "masses" in self.nominal_params and hasattr(self.engine, "set_link_masses"):
            self.engine.set_link_masses(self.nominal_params["masses"])

        if "friction" in self.nominal_params and hasattr(
            self.engine, "set_friction_coefficients"
        ):
            self.engine.set_friction_coefficients(self.nominal_params["friction"])

        if "damping" in self.nominal_params and hasattr(
            self.engine, "set_joint_damping"
        ):
            self.engine.set_joint_damping(self.nominal_params["damping"])

        if "motor_strength" in self.nominal_params and hasattr(
            self.engine, "set_motor_strength"
        ):
            self.engine.set_motor_strength(self.nominal_params["motor_strength"])

        if "gravity" in self.nominal_params and hasattr(self.engine, "set_gravity"):
            self.engine.set_gravity(self.nominal_params["gravity"])

        self._action_delay = 0
        self._observation_delay = 0
        self._action_buffer = []
        self._observation_buffer = []
        self._current_randomization = {}

    def apply_action_with_delay(
        self, action: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply action delay.

        Args:
            action: Current action to apply.

        Returns:
            Delayed action to actually apply.
        """
        if not self.config.randomize_delays or self._action_delay == 0:
            return self._apply_action_noise(action)

        self._action_buffer.append(action)

        if len(self._action_buffer) <= self._action_delay:
            # Return zero action during initial delay
            return np.zeros_like(action)

        delayed_action = self._action_buffer.pop(0)
        return self._apply_action_noise(delayed_action)

    def _apply_action_noise(
        self, action: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply action noise.

        Args:
            action: Action to add noise to.

        Returns:
            Noisy action.
        """
        if not self.config.randomize_noise or self.config.action_noise_std == 0:
            return action

        noise = self._rng.normal(0, self.config.action_noise_std, action.shape)
        return action + noise

    def get_observation_with_delay(
        self, observation: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply observation delay and noise.

        Args:
            observation: Current observation.

        Returns:
            Delayed and noisy observation.
        """
        if not self.config.randomize_delays or self._observation_delay == 0:
            return self._apply_observation_noise(observation)

        self._observation_buffer.append(observation)

        if len(self._observation_buffer) <= self._observation_delay:
            # Return first observation during initial delay
            return self._apply_observation_noise(self._observation_buffer[0])

        delayed_obs = self._observation_buffer.pop(0)
        return self._apply_observation_noise(delayed_obs)

    def _apply_observation_noise(
        self, observation: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Apply observation noise.

        Args:
            observation: Observation to add noise to.

        Returns:
            Noisy observation.
        """
        if not self.config.randomize_noise or self.config.observation_noise_std == 0:
            return observation

        noise = self._rng.normal(
            0, self.config.observation_noise_std, observation.shape
        )
        return observation + noise

    def get_current_randomization(self) -> dict[str, Any]:
        """Get current randomization values.

        Returns:
            Dictionary of current randomization parameters.
        """
        return self._current_randomization.copy()

    def sample_randomization_batch(
        self, batch_size: int
    ) -> list[dict[str, Any]]:
        """Sample a batch of randomization configurations.

        Useful for parallel environment training.

        Args:
            batch_size: Number of configurations to sample.

        Returns:
            List of randomization dictionaries.
        """
        configs = []
        for i in range(batch_size):
            config = self.randomize(seed=self._rng.integers(0, 2**31))
            configs.append(config)
            self.reset_to_nominal()
        return configs
