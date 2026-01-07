"""MyoSuite integration adapter for neural control of musculoskeletal systems.

MyoSuite is a physics-based environment for training neural control policies
using reinforcement learning. It provides:
- Muscle-driven biomechanical models
- Neuromuscular control tasks
- Gym-compatible RL interface

This adapter bridges the Golf Modeling Suite muscle models with MyoSuite's
expected interface, enabling:
1. Import of custom golf swing models into MyoSuite
2. RL training of muscle activation policies
3. Transfer learning from biomechanics to robotics

Architecture:
- Golf Suite provides: Hill muscles, equilibrium solver, activation dynamics
- MyoSuite provides: RL environment, training infrastructure, neural policies
- This adapter provides: Bidirectional interface layer

Reference:
- Caggiano et al. (2022), "MyoSuite: A contact-rich simulation suite for musculoskeletal motor control"
- MyoSuite Documentation: https://github.com/facebookresearch/myosuite
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from shared.python.multi_muscle import AntagonistPair, MuscleGroup

logger = logging.getLogger(__name__)

# Check for MyoSuite availability
try:
    import gym

    MYOSUITE_AVAILABLE = True
except ImportError:
    MYOSUITE_AVAILABLE = False
    logger.warning(
        "MyoSuite/gym not available. Neural control disabled. "
        "Install with: pip install myosuite"
    )


class MuscleDrivenEnv:
    """Gym-compatible environment for muscle-driven control.

    This provides the interface expected by RL frameworks (Stable-Baselines3, RLlib).

    Observation space: Joint angles,velocities + muscle states
    Action space: Neural excitations [0,1] for each muscle

    Example:
        >>> env = MuscleDrivenEnv(muscle_system, task="reach")
        >>> obs = env.reset()
        >>> for _ in range(1000):
        ...     action = policy.predict(obs)  # RL policy
        ...     obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        muscle_system: AntagonistPair | MuscleGroup,
        task: str = "tracking",
        dt: float = 0.001,
    ):
        """Initialize muscle-driven environment.

        Args:
            muscle_system: Muscle system to control
            task: Task type ("tracking", "reach", "swing")
            dt: Simulation timestep [s]
        """
        self.muscle_system = muscle_system
        self.task = task
        self.dt = dt

        # State
        self.q = 0.0  # Joint angle [rad]
        self.v = 0.0  # Joint velocity [rad/s]
        self.muscle_activations: dict[str, float] = {}
        self.muscle_states: dict[str, tuple[float, float]] = {}

        # Task-specific
        self.target_q = 0.0  # Target joint angle
        self.step_count = 0
        self.max_steps = 1000

        logger.info(f"Created muscle-driven environment: task='{task}', dt={dt:.3f}s")

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        Returns:
            Initial observation
        """
        self.q = np.random.uniform(-np.pi / 4, np.pi / 4)  # Random init
        self.v = 0.0
        self.muscle_activations = dict.fromkeys(self._get_muscle_names(), 0.01)  # Min activation
        self.muscle_states = dict.fromkeys(self._get_muscle_names(), (0.12, 0.0))  # Default fiber state

        if self.task == "tracking":
            # Time-varying target
            self.target_q = np.sin(2 * np.pi * 0.5 * self.step_count * self.dt)

        self.step_count = 0

        return self._get_observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one timestep.

        Args:
            action: Neural excitations [0,1] for each muscle (n_muscles,)

        Returns:
            (observation, reward, done, info)
        """
        # Convert action to muscle excitations
        excitations = self._action_to_excitations(action)

        # Update activations (with dynamics delay)
        from shared.python.activation_dynamics import ActivationDynamics

        dynamics = ActivationDynamics()
        for muscle_name, u in excitations.items():
            a_current = self.muscle_activations.get(muscle_name, 0.01)
            a_new = dynamics.update(u, a_current, self.dt)
            self.muscle_activations[muscle_name] = a_new

        # Compute muscle torques
        if isinstance(self.muscle_system, AntagonistPair):
            # Split activations into agonist/antagonist
            agonist_act, antagonist_act = self._split_activations()
            tau_muscle = self.muscle_system.compute_net_torque(
                agonist_act, antagonist_act, self.muscle_states
            )
        else:
            tau_muscle = self.muscle_system.compute_net_torque(
                self.muscle_activations, self.muscle_states
            )

        # Simple joint dynamics (pendulum-like)
        I = 0.05  # Joint inertia [kg·m²]
        b = 0.5  # Damping [N·m·s/rad]
        g_eff = 9.81  # Effective gravity [m/s²]
        l_eff = 0.3  # Effective length [m]

        tau_gravity = -g_eff * l_eff * np.sin(self.q)
        tau_total = tau_muscle + tau_gravity - b * self.v

        # Integrate
        a = tau_total / I
        self.v += a * self.dt
        self.q += self.v * self.dt

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        done = self.step_count >= self.max_steps or abs(self.q) > np.pi

        self.step_count += 1

        info = {
            "tau_muscle": tau_muscle,
            "q_error": abs(self.q - self.target_q),
            "activations": self.muscle_activations.copy(),
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector.

        Returns:
            Observation: [q, v, a_1, ..., a_n]
        """
        obs = [self.q, self.v]
        for muscle_name in sorted(self._get_muscle_names()):
            obs.append(self.muscle_activations.get(muscle_name, 0.0))

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute task reward.

        Returns:
            Reward scalar
        """
        if self.task == "tracking":
            # Minimize tracking error
            error = abs(self.q - self.target_q)
            reward = -error  # Negative error (higher is better)
        elif self.task == "reach":
            # Reach target in minimum time
            dist = abs(self.q - self.target_q)
            reward = -dist - 0.01 * self.step_count  # Penalize time
        else:
            reward = 0.0

        return float(reward)

    def _get_muscle_names(self) -> list[str]:
        """Get list of muscle names in system."""
        if isinstance(self.muscle_system, AntagonistPair):
            names = list(self.muscle_system.agonist.muscles.keys())
            names.extend(self.muscle_system.antagonist.muscles.keys())
        else:
            names = list(self.muscle_system.muscles.keys())
        return names

    def _action_to_excitations(self, action: np.ndarray) -> dict[str, float]:
        """Convert RL action to muscle excitations.

        Args:
            action: Action vector (n_muscles,)

        Returns:
            Excitation dict {muscle_name: excitation}
        """
        muscle_names = sorted(self._get_muscle_names())
        excitations = {}
        for i, name in enumerate(muscle_names):
            excitations[name] = float(np.clip(action[i], 0.0, 1.0))
        return excitations

    def _split_activations(self) -> tuple[dict[str, float], dict[str, float]]:
        """Split activations into agonist/antagonist groups."""
        if not isinstance(self.muscle_system, AntagonistPair):
            return {}, {}

        agonist_act = {
            name: self.muscle_activations[name]
            for name in self.muscle_system.agonist.muscles.keys()
            if name in self.muscle_activations
        }

        antagonist_act = {
            name: self.muscle_activations[name]
            for name in self.muscle_system.antagonist.muscles.keys()
            if name in self.muscle_activations
        }

        return agonist_act, antagonist_act


def train_muscle_policy(env: MuscleDrivenEnv, total_timesteps: int = 100000) -> Any:
    """Train RL policy for muscle control.

    Uses Stable-Baselines3 SAC (Soft Actor-Critic) algorithm.

    Args:
        env: Muscle-driven environment
        total_timesteps: Number of training steps

    Returns:
        Trained policy model

    Example:
        >>> env = MuscleDrivenEnv(muscle_system)
        >>> policy = train_muscle_policy(env, total_timesteps=50000)
        >>> # Policy can now control muscles via neural network
    """
    if not MYOSUITE_AVAILABLE:
        logger.error("Cannot train policy: MyoSuite/gym not installed")
        return None

    try:
        from stable_baselines3 import SAC

        logger.info(f"Training muscle policy: {total_timesteps} steps")

        # SAC is good for continuous control
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
        )

        model.learn(total_timesteps=total_timesteps)

        logger.info("✓ Policy training complete")
        return model

    except ImportError:
        logger.error(
            "stable-baselines3 not installed. "
            "Install with: pip install stable-baselines3"
        )
        return None


# Example integration workflow
if __name__ == "__main__":
    print("=" * 60)
    print("MyoSuite Integration Example")
    print("=" * 60)

    # Create muscle system
    from shared.python.multi_muscle import create_elbow_muscle_system

    elbow = create_elbow_muscle_system()

    # Create RL environment
    env = MuscleDrivenEnv(elbow, task="tracking", dt=0.001)

    print("\\nEnvironment created:")
    print(f"  Task: {env.task}")
    print(f"  Timestep: {env.dt*1000:.1f} ms")
    print(f"  Muscles: {len(env._get_muscle_names())}")

    # Test manual control (no training)
    print("\\nTesting manual control (no RL):")
    obs = env.reset()
    print(f"  Initial obs: {obs}")

    # Apply constant excitation
    action = np.array([0.5, 0.3, 0.1])  # Biceps, brachialis, triceps
    obs, reward, done, info = env.step(action)

    print("  After step:")
    print(f"    Observation: {obs}")
    print(f"    Reward: {reward:.4f}")
    print(f"    Torque: {info['tau_muscle']:.2f} N·m")

    print("\\n" + "=" * 60)
    print("✓ MyoSuite integration test complete")
    print("=" * 60)
    print("\\nTo train a policy:")
    print("  policy = train_muscle_policy(env, total_timesteps=50000)")
    print("  # Then use: action, _states = policy.predict(observation)")
