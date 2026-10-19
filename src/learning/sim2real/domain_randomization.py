"""Domain Randomization for sim-to-real transfer."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.learning.sim2real.system_identification import UnsupportedParameterError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.engines.protocols import PhysicsEngineProtocol


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization.

    Randomization ranges are specified as (min_multiplier, max_multiplier)
    relative to nominal values, except ``gravity_range`` which is magnitude
    in m/s².

    Attributes:
        mass_range: Range for link mass randomization.
        friction_range: Range for friction coefficient.
        damping_range: Range for joint damping.
        motor_strength_range: Range for motor strength and actuator gains.
        action_delay_range: Range for action delay (timesteps).
        observation_delay_range: Range for observation delay.
        observation_noise_std: Std dev of observation noise.
        action_noise_std: Std dev of action noise.
        gravity_range: Range for gravity magnitude.
    """

    # Dynamics parameters
    mass_range: tuple[float, float] = (0.8, 1.2)
    friction_range: tuple[float, float] = (0.5, 1.5)
    damping_range: tuple[float, float] = (0.8, 1.2)
    motor_strength_range: tuple[float, float] = (0.9, 1.1)

    # Delays and noise
    action_delay_range: tuple[int, int] = (0, 3)
    observation_delay_range: tuple[int, int] = (0, 2)
    observation_noise_std: float = 0.01
    action_noise_std: float = 0.01

    # Environment
    gravity_range: tuple[float, float] = (9.5, 10.1)

    # Which parameters to randomize
    randomize_mass: bool = True
    randomize_friction: bool = True
    randomize_damping: bool = True
    randomize_motor: bool = True
    randomize_gravity: bool = True
    randomize_delays: bool = True
    randomize_noise: bool = True


#: cache key -> (getter name, setter name, config flag attribute)
_PHYSICS_SURFACE: dict[str, tuple[str, str, str]] = {
    "masses": ("get_link_masses", "set_link_masses", "randomize_mass"),
    "friction": (
        "get_friction_coefficients",
        "set_friction_coefficients",
        "randomize_friction",
    ),
    "damping": ("get_joint_damping", "set_joint_damping", "randomize_damping"),
    "motor_strength": ("get_motor_strength", "set_motor_strength", "randomize_motor"),
    "actuator_gains": ("get_actuator_gains", "set_actuator_gains", "randomize_motor"),
    "gravity": ("get_gravity", "set_gravity", "randomize_gravity"),
}


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
        *,
        seed: int | None = None,
    ) -> None:
        """Initialize domain randomizer.

        Args:
            engine: Physics engine instance.
            config: Randomization configuration.
            seed: Optional seed for the internal RNG (issue #13014).
        """
        if engine is None:
            raise ValueError("engine must be provided")
        self.engine = engine
        self.config = config or DomainRandomizationConfig()
        self.nominal_params = self._store_nominal_params()
        self._rng = np.random.default_rng(seed)
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
        params: dict[str, Any] = {}
        for key, (getter, _setter, _flag) in _PHYSICS_SURFACE.items():
            fn = getattr(self.engine, getter, None)
            if fn is None:
                continue
            params[key] = np.asarray(fn(), dtype=float).copy()
        return params

    def _requested_physics_keys(self) -> list[str]:
        keys: list[str] = []
        for key, (_getter, _setter, flag_name) in _PHYSICS_SURFACE.items():
            if getattr(self.config, flag_name):
                keys.append(key)
        return keys

    def _setter_is_effective(self, key: str, getter: str, setter: str) -> bool:
        """Return True when ``setter`` measurably changes ``getter`` output."""
        if key not in self.nominal_params:
            return False
        nominal = np.asarray(self.nominal_params[key], dtype=float)
        if nominal.size == 0:
            return False
        probe = nominal * 1.5 + 0.5
        try:
            getattr(self.engine, setter)(probe)
            observed = np.asarray(getattr(self.engine, getter)(), dtype=float)
        except (TypeError, ValueError, NotImplementedError, AttributeError):
            return False
        finally:
            with contextlib.suppress(
                TypeError, ValueError, NotImplementedError, AttributeError
            ):
                getattr(self.engine, setter)(nominal.copy())
        return bool(
            observed.shape == probe.shape and not np.allclose(observed, nominal)
        )

    def _validate_physics_support(self) -> None:
        """Refuse silent no-ops when physics randomization was requested (#13014)."""
        missing: list[str] = []
        for key in self._requested_physics_keys():
            hooks = _PHYSICS_SURFACE.get(key)
            if hooks is None:
                continue
            getter_name, setter_name, _flag = hooks
            if key not in self.nominal_params:
                missing.append(f"{key}: engine exposes no {getter_name}()")
                continue
            if not self._setter_is_effective(key, getter_name, setter_name):
                missing.append(
                    f"{key}: requires working {getter_name}()/{setter_name}()"
                )
        if not missing:
            return
        raise UnsupportedParameterError(
            "Cannot randomize physics on "
            f"{type(self.engine).__name__}: " + "; ".join(missing) + ". "
            "Training would believe dynamics were randomized when they were not "
            "(issue #13014)."
        )

    def randomize(self, seed: int | None = None) -> dict[str, Any]:  # noqa: C901
        """Randomize simulation parameters.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Dictionary of applied randomization values.

        Raises:
            UnsupportedParameterError: When a requested physics parameter cannot
                be applied on this engine.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._validate_physics_support()

        randomization: dict[str, Any] = {}

        # Randomize mass
        if self.config.randomize_mass and "masses" in self.nominal_params:
            scale = self._rng.uniform(*self.config.mass_range)
            new_masses = self.nominal_params["masses"] * scale
            self.engine.set_link_masses(new_masses)
            randomization["mass_scale"] = scale

        # Randomize friction
        if self.config.randomize_friction and "friction" in self.nominal_params:
            scale = self._rng.uniform(*self.config.friction_range)
            new_friction = self.nominal_params["friction"] * scale
            self.engine.set_friction_coefficients(new_friction)
            randomization["friction_scale"] = scale

        # Randomize damping
        if self.config.randomize_damping and "damping" in self.nominal_params:
            scale = self._rng.uniform(*self.config.damping_range)
            new_damping = self.nominal_params["damping"] * scale
            self.engine.set_joint_damping(new_damping)
            randomization["damping_scale"] = scale

        # Randomize motor strength
        if self.config.randomize_motor and "motor_strength" in self.nominal_params:
            scale = self._rng.uniform(*self.config.motor_strength_range)
            new_strength = self.nominal_params["motor_strength"] * scale
            self.engine.set_motor_strength(new_strength)
            randomization["motor_scale"] = scale

        if self.config.randomize_motor and "actuator_gains" in self.nominal_params:
            scale = self._rng.uniform(*self.config.motor_strength_range)
            new_gains = self.nominal_params["actuator_gains"] * scale
            self.engine.set_actuator_gains(new_gains)
            randomization["actuator_gain_scale"] = scale

        # Randomize gravity magnitude along the engine's nominal axis
        if self.config.randomize_gravity and "gravity" in self.nominal_params:
            g_mag = self._rng.uniform(*self.config.gravity_range)
            nominal_g = np.asarray(self.nominal_params["gravity"], dtype=float)
            axis = int(np.argmax(np.abs(nominal_g)))
            sign = float(np.sign(nominal_g[axis])) if nominal_g[axis] != 0 else -1.0
            gravity = nominal_g.copy()
            gravity[axis] = sign * g_mag
            self.engine.set_gravity(gravity)
            randomization["gravity"] = g_mag

        # Randomize delays
        if self.config.randomize_delays:
            self._action_delay = int(
                self._rng.integers(
                    self.config.action_delay_range[0],
                    self.config.action_delay_range[1] + 1,
                )
            )
            self._observation_delay = int(
                self._rng.integers(
                    self.config.observation_delay_range[0],
                    self.config.observation_delay_range[1] + 1,
                )
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
        if "masses" in self.nominal_params:
            self.engine.set_link_masses(self.nominal_params["masses"])

        if "friction" in self.nominal_params:
            self.engine.set_friction_coefficients(self.nominal_params["friction"])

        if "damping" in self.nominal_params:
            self.engine.set_joint_damping(self.nominal_params["damping"])

        if "motor_strength" in self.nominal_params:
            self.engine.set_motor_strength(self.nominal_params["motor_strength"])

        if "actuator_gains" in self.nominal_params:
            self.engine.set_actuator_gains(self.nominal_params["actuator_gains"])

        if "gravity" in self.nominal_params:
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
        if action is None:
            raise ValueError("action must be provided")
        if not self.config.randomize_delays or self._action_delay == 0:
            return self._apply_action_noise(action)

        self._action_buffer.append(action)

        if len(self._action_buffer) <= self._action_delay:
            # Return zero action during initial delay
            return np.zeros_like(action)

        delayed_action = self._action_buffer.pop(0)
        return self._apply_action_noise(delayed_action)

    def _apply_action_noise(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply action noise.

        Args:
            action: Action to add noise to.

        Returns:
            Noisy action.
        """
        if action is None:
            raise ValueError("action must be provided")
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
        if observation is None:
            raise ValueError("observation must be provided")
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
        if observation is None:
            raise ValueError("observation must be provided")
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

    def sample_randomization_batch(self, batch_size: int) -> list[dict[str, Any]]:
        """Sample a batch of randomization configurations.

        Useful for parallel environment training.

        Args:
            batch_size: Number of configurations to sample.

        Returns:
            List of randomization dictionaries.
        """
        if batch_size is None:
            raise ValueError("batch_size must be provided")
        configs = []
        for _i in range(batch_size):
            config = self.randomize(seed=int(self._rng.integers(0, 2**31)))
            configs.append(config)
            self.reset_to_nominal()
        return configs
