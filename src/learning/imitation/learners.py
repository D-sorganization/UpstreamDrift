"""Imitation learning algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None

from src.learning.imitation.dataset import Demonstration, DemonstrationDataset


@dataclass
class TrainingConfig:
    """Configuration for imitation learning training.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        weight_decay: L2 regularization weight.
        hidden_sizes: Hidden layer sizes for neural network.
        activation: Activation function name.
        dropout: Dropout probability.
    """

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dropout: float = 0.0


class ImitationLearner(ABC):
    """Base class for imitation learning algorithms.

    Subclasses implement specific algorithms like behavior cloning,
    DAgger, or GAIL.

    Attributes:
        observation_dim: Dimension of observation space.
        action_dim: Dimension of action space.
        config: Training configuration.
        device: Compute device (cpu, cuda).
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: TrainingConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize imitation learner.

        Args:
            observation_dim: Dimension of observation space.
            action_dim: Dimension of action space.
            config: Training configuration.
            device: Compute device.
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config or TrainingConfig()
        self.device = device
        self._policy: Any = None  # Neural network policy
        self._training_history: dict[str, list[float]] = {}

    @classmethod
    def from_spaces(
        cls,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: TrainingConfig | None = None,
        device: str = "cpu",
    ) -> "ImitationLearner":
        """Create learner from Gymnasium spaces.

        Args:
            observation_space: Observation space.
            action_space: Action space.
            config: Training configuration.
            device: Compute device.

        Returns:
            Imitation learner instance.
        """
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        return cls(obs_dim, act_dim, config, device)

    @abstractmethod
    def train(
        self,
        dataset: DemonstrationDataset,
        validation_split: float = 0.1,
    ) -> dict[str, list[float]]:
        """Train policy on demonstrations.

        Args:
            dataset: Demonstration dataset.
            validation_split: Fraction of data for validation.

        Returns:
            Training history with loss curves.
        """
        pass

    @abstractmethod
    def predict(
        self,
        observation: NDArray[np.floating],
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Predict action for observation.

        Args:
            observation: Current observation.
            deterministic: If True, return mean action.

        Returns:
            Predicted action.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save trained policy.

        Args:
            path: Path to save file.
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load trained policy.

        Args:
            path: Path to load file.
        """
        pass

    def get_training_history(self) -> dict[str, list[float]]:
        """Get training history.

        Returns:
            Dictionary with training metrics over epochs.
        """
        return self._training_history.copy()


class BehaviorCloning(ImitationLearner):
    """Behavior Cloning via supervised learning.

    Learns a policy that maps states to actions using supervised
    regression on demonstration data. Simple but can suffer from
    distribution shift.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: TrainingConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize behavior cloning learner."""
        super().__init__(observation_dim, action_dim, config, device)
        self._build_policy()

    def _build_policy(self) -> None:
        """Build the neural network policy."""
        # Build a simple MLP policy
        # In production, would use PyTorch/JAX
        # Here we implement a simple numpy-based MLP for demonstration
        layers = []
        input_dim = self.observation_dim

        for hidden_dim in self.config.hidden_sizes:
            layers.append({
                "W": np.random.randn(input_dim, hidden_dim) * 0.01,
                "b": np.zeros(hidden_dim),
            })
            input_dim = hidden_dim

        # Output layer
        layers.append({
            "W": np.random.randn(input_dim, self.action_dim) * 0.01,
            "b": np.zeros(self.action_dim),
        })

        self._policy = layers

    def _forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass through network.

        Args:
            x: Input observations.

        Returns:
            Predicted actions.
        """
        for i, layer in enumerate(self._policy):
            x = x @ layer["W"] + layer["b"]
            # ReLU activation for hidden layers
            if i < len(self._policy) - 1:
                x = np.maximum(0, x)
        return x

    def _compute_loss(
        self,
        observations: NDArray[np.floating],
        actions: NDArray[np.floating],
    ) -> float:
        """Compute MSE loss.

        Args:
            observations: Batch of observations.
            actions: Batch of target actions.

        Returns:
            Mean squared error loss.
        """
        predictions = self._forward(observations)
        return float(np.mean((predictions - actions) ** 2))

    def _backward(
        self,
        observations: NDArray[np.floating],
        actions: NDArray[np.floating],
    ) -> list[dict[str, NDArray[np.floating]]]:
        """Compute gradients via backpropagation.

        Args:
            observations: Batch of observations.
            actions: Batch of target actions.

        Returns:
            List of gradient dictionaries for each layer.
        """
        batch_size = len(observations)

        # Forward pass with caching
        activations = [observations]
        x = observations
        for i, layer in enumerate(self._policy):
            z = x @ layer["W"] + layer["b"]
            if i < len(self._policy) - 1:
                x = np.maximum(0, z)  # ReLU
            else:
                x = z  # Linear output
            activations.append(x)

        # Backward pass
        gradients = []
        predictions = activations[-1]
        delta = 2 * (predictions - actions) / batch_size  # MSE gradient

        for i in range(len(self._policy) - 1, -1, -1):
            layer = self._policy[i]
            a = activations[i]

            grad_W = a.T @ delta
            grad_b = delta.sum(axis=0)
            gradients.insert(0, {"W": grad_W, "b": grad_b})

            if i > 0:
                delta = delta @ layer["W"].T
                # ReLU gradient
                delta = delta * (activations[i] > 0)

        return gradients

    def train(
        self,
        dataset: DemonstrationDataset,
        validation_split: float = 0.1,
    ) -> dict[str, list[float]]:
        """Train behavior cloning policy.

        Args:
            dataset: Demonstration dataset.
            validation_split: Fraction for validation.

        Returns:
            Training history.
        """
        # Get training data
        observations, actions = dataset.to_state_action_pairs()

        if len(observations) == 0:
            raise ValueError("Dataset has no state-action pairs")

        # Split data
        n = len(observations)
        n_val = int(n * validation_split)
        indices = np.random.permutation(n)

        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        train_obs = observations[train_idx]
        train_act = actions[train_idx]
        val_obs = observations[val_idx] if n_val > 0 else train_obs[:100]
        val_act = actions[val_idx] if n_val > 0 else train_act[:100]

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        lr = self.config.learning_rate

        for epoch in range(self.config.epochs):
            # Shuffle training data
            perm = np.random.permutation(len(train_obs))
            train_obs = train_obs[perm]
            train_act = train_act[perm]

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_obs), self.config.batch_size):
                batch_obs = train_obs[i : i + self.config.batch_size]
                batch_act = train_act[i : i + self.config.batch_size]

                # Compute gradients
                gradients = self._backward(batch_obs, batch_act)

                # Update weights
                for layer, grad in zip(self._policy, gradients):
                    layer["W"] -= lr * (
                        grad["W"] + self.config.weight_decay * layer["W"]
                    )
                    layer["b"] -= lr * grad["b"]

                epoch_loss += self._compute_loss(batch_obs, batch_act)
                n_batches += 1

            # Record metrics
            train_loss = epoch_loss / n_batches
            val_loss = self._compute_loss(val_obs, val_act)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

        self._training_history = history
        return history

    def predict(
        self,
        observation: NDArray[np.floating],
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Predict action from observation.

        Args:
            observation: Current observation.
            deterministic: If True, return deterministic action.

        Returns:
            Predicted action.
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        action = self._forward(observation)

        if observation.shape[0] == 1:
            action = action.flatten()

        return action

    def save(self, path: str | Path) -> None:
        """Save policy to disk.

        Args:
            path: Path to save file.
        """
        path = Path(path)
        data = {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "hidden_sizes": self.config.hidden_sizes,
            },
            "layers": [
                {"W": layer["W"].tolist(), "b": layer["b"].tolist()}
                for layer in self._policy
            ],
        }
        np.savez(path, **{k: np.array(v, dtype=object) for k, v in data.items()})

    def load(self, path: str | Path) -> None:
        """Load policy from disk.

        Args:
            path: Path to load file.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        self.observation_dim = int(data["observation_dim"])
        self.action_dim = int(data["action_dim"])

        layers_data = data["layers"].item()
        self._policy = [
            {"W": np.array(layer["W"]), "b": np.array(layer["b"])}
            for layer in layers_data
        ]


class DAgger(ImitationLearner):
    """Dataset Aggregation with expert queries.

    DAgger iteratively collects data using the current policy
    but labels with expert actions, addressing distribution shift.

    Requires access to an expert policy during training.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: TrainingConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize DAgger learner."""
        super().__init__(observation_dim, action_dim, config, device)
        self._bc = BehaviorCloning(observation_dim, action_dim, config, device)
        self._aggregated_dataset: DemonstrationDataset | None = None

    def train(
        self,
        dataset: DemonstrationDataset,
        validation_split: float = 0.1,
    ) -> dict[str, list[float]]:
        """Train initial policy with behavior cloning.

        For DAgger iterations, use train_online().

        Args:
            dataset: Initial demonstration dataset.
            validation_split: Fraction for validation.

        Returns:
            Training history.
        """
        self._aggregated_dataset = dataset
        return self._bc.train(dataset, validation_split)

    def train_online(
        self,
        env: Any,  # RoboticsGymEnv
        expert: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        iterations: int = 10,
        trajectories_per_iter: int = 10,
        max_steps: int = 500,
        beta_schedule: str = "linear",
    ) -> dict[str, Any]:
        """Online training with expert intervention.

        Args:
            env: Gymnasium environment.
            expert: Expert policy function.
            iterations: Number of DAgger iterations.
            trajectories_per_iter: Trajectories per iteration.
            max_steps: Max steps per trajectory.
            beta_schedule: Schedule for mixing policy and expert.

        Returns:
            Training results.
        """
        if self._aggregated_dataset is None:
            raise ValueError("Must call train() first with initial dataset")

        results = {
            "iteration_rewards": [],
            "dataset_size": [],
        }

        for iteration in range(iterations):
            # Compute beta (probability of using expert)
            if beta_schedule == "linear":
                beta = 1.0 - iteration / iterations
            else:
                beta = 0.5 ** iteration

            # Collect trajectories
            new_demos = []
            iteration_rewards = []

            for _ in range(trajectories_per_iter):
                obs, info = env.reset()
                demo_timestamps = [0.0]
                demo_positions = [obs[: obs.shape[0] // 2]]
                demo_velocities = [obs[obs.shape[0] // 2 :]]
                demo_actions = []

                total_reward = 0.0
                step = 0

                while step < max_steps:
                    # Get policy action
                    policy_action = self.predict(obs)

                    # Get expert action for labeling
                    expert_action = expert(obs)

                    # Use expert action for dataset
                    demo_actions.append(expert_action)

                    # Execute action (mix policy and expert)
                    if np.random.random() < beta:
                        action = expert_action
                    else:
                        action = policy_action

                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    step += 1

                    demo_timestamps.append(step * 0.01)
                    demo_positions.append(obs[: obs.shape[0] // 2])
                    demo_velocities.append(obs[obs.shape[0] // 2 :])

                    if terminated or truncated:
                        break

                # Create demonstration
                demo = Demonstration(
                    timestamps=np.array(demo_timestamps[:-1]),
                    joint_positions=np.array(demo_positions[:-1]),
                    joint_velocities=np.array(demo_velocities[:-1]),
                    actions=np.array(demo_actions),
                    source="dagger",
                    success=not terminated,
                )
                new_demos.append(demo)
                iteration_rewards.append(total_reward)

            # Aggregate dataset
            self._aggregated_dataset.extend(new_demos)

            # Retrain on aggregated dataset
            self._bc.train(self._aggregated_dataset)

            # Record results
            results["iteration_rewards"].append(np.mean(iteration_rewards))
            results["dataset_size"].append(len(self._aggregated_dataset))

        return results

    def predict(
        self,
        observation: NDArray[np.floating],
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Predict action using trained policy.

        Args:
            observation: Current observation.
            deterministic: If True, return deterministic action.

        Returns:
            Predicted action.
        """
        return self._bc.predict(observation, deterministic)

    def save(self, path: str | Path) -> None:
        """Save policy."""
        self._bc.save(path)

    def load(self, path: str | Path) -> None:
        """Load policy."""
        self._bc.load(path)


class GAIL(ImitationLearner):
    """Generative Adversarial Imitation Learning.

    GAIL uses a discriminator to distinguish between expert and
    policy trajectories, training the policy to fool the discriminator.

    This is a simplified implementation - full GAIL requires
    integration with RL algorithms like PPO.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: TrainingConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize GAIL learner."""
        super().__init__(observation_dim, action_dim, config, device)
        self._policy = None
        self._discriminator = None
        self._build_networks()

    def _build_networks(self) -> None:
        """Build policy and discriminator networks."""
        # Build simple MLP policy
        policy_layers = []
        input_dim = self.observation_dim

        for hidden_dim in self.config.hidden_sizes:
            policy_layers.append({
                "W": np.random.randn(input_dim, hidden_dim) * 0.01,
                "b": np.zeros(hidden_dim),
            })
            input_dim = hidden_dim

        policy_layers.append({
            "W": np.random.randn(input_dim, self.action_dim) * 0.01,
            "b": np.zeros(self.action_dim),
        })
        self._policy = policy_layers

        # Build discriminator (state-action -> [0, 1])
        disc_layers = []
        input_dim = self.observation_dim + self.action_dim

        for hidden_dim in self.config.hidden_sizes:
            disc_layers.append({
                "W": np.random.randn(input_dim, hidden_dim) * 0.01,
                "b": np.zeros(hidden_dim),
            })
            input_dim = hidden_dim

        disc_layers.append({
            "W": np.random.randn(input_dim, 1) * 0.01,
            "b": np.zeros(1),
        })
        self._discriminator = disc_layers

    def _forward_policy(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass through policy network."""
        for i, layer in enumerate(self._policy):
            x = x @ layer["W"] + layer["b"]
            if i < len(self._policy) - 1:
                x = np.maximum(0, x)  # ReLU
        return x

    def _forward_discriminator(
        self, state: NDArray[np.floating], action: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Forward pass through discriminator."""
        x = np.concatenate([state, action], axis=-1)
        for i, layer in enumerate(self._discriminator):
            x = x @ layer["W"] + layer["b"]
            if i < len(self._discriminator) - 1:
                x = np.maximum(0, x)  # ReLU
            else:
                x = 1 / (1 + np.exp(-x))  # Sigmoid
        return x

    def train(
        self,
        dataset: DemonstrationDataset,
        validation_split: float = 0.1,
    ) -> dict[str, list[float]]:
        """Train GAIL.

        Note: This is a simplified version. Full GAIL training
        requires environment interaction and RL algorithm integration.

        Args:
            dataset: Expert demonstration dataset.
            validation_split: Fraction for validation.

        Returns:
            Training history.
        """
        # Get expert data
        expert_states, expert_actions = dataset.to_state_action_pairs()

        if len(expert_states) == 0:
            raise ValueError("Dataset has no state-action pairs")

        history = {"discriminator_loss": [], "policy_loss": []}
        lr = self.config.learning_rate

        for epoch in range(self.config.epochs):
            # Generate policy data (self-play would go here)
            # For simplicity, we just use noise-perturbed expert data
            noise = np.random.randn(*expert_states.shape) * 0.1
            policy_states = expert_states + noise
            policy_actions = self._forward_policy(policy_states)

            # Train discriminator
            expert_preds = self._forward_discriminator(expert_states, expert_actions)
            policy_preds = self._forward_discriminator(policy_states, policy_actions)

            # Binary cross entropy
            eps = 1e-8
            disc_loss = -np.mean(
                np.log(expert_preds + eps) + np.log(1 - policy_preds + eps)
            )

            # Update discriminator (simplified gradient)
            expert_grad = expert_preds - 1  # gradient towards 1
            policy_grad = policy_preds  # gradient towards 0

            for i, layer in enumerate(self._discriminator):
                # Simplified update
                layer["W"] -= lr * 0.01 * layer["W"]
                layer["b"] -= lr * 0.01 * layer["b"]

            # Policy reward is discriminator output
            policy_reward = -np.log(1 - policy_preds + eps)
            policy_loss = -np.mean(policy_reward)

            history["discriminator_loss"].append(float(disc_loss))
            history["policy_loss"].append(float(policy_loss))

        self._training_history = history
        return history

    def predict(
        self,
        observation: NDArray[np.floating],
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Predict action from observation.

        Args:
            observation: Current observation.
            deterministic: If True, return deterministic action.

        Returns:
            Predicted action.
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        action = self._forward_policy(observation)

        if not deterministic:
            action = action + np.random.randn(*action.shape) * 0.1

        if observation.shape[0] == 1:
            action = action.flatten()

        return action

    def get_reward(
        self,
        state: NDArray[np.floating],
        action: NDArray[np.floating],
    ) -> float:
        """Get GAIL reward for state-action pair.

        Args:
            state: Current state.
            action: Taken action.

        Returns:
            GAIL reward value.
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)

        disc_output = self._forward_discriminator(state, action)
        # Reward is -log(1 - D(s,a))
        return float(-np.log(1 - disc_output + 1e-8))

    def save(self, path: str | Path) -> None:
        """Save GAIL networks."""
        path = Path(path)
        data = {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "policy": [
                {"W": layer["W"].tolist(), "b": layer["b"].tolist()}
                for layer in self._policy
            ],
            "discriminator": [
                {"W": layer["W"].tolist(), "b": layer["b"].tolist()}
                for layer in self._discriminator
            ],
        }
        np.savez(path, **{k: np.array(v, dtype=object) for k, v in data.items()})

    def load(self, path: str | Path) -> None:
        """Load GAIL networks."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)

        self.observation_dim = int(data["observation_dim"])
        self.action_dim = int(data["action_dim"])

        policy_data = data["policy"].item()
        self._policy = [
            {"W": np.array(layer["W"]), "b": np.array(layer["b"])}
            for layer in policy_data
        ]

        disc_data = data["discriminator"].item()
        self._discriminator = [
            {"W": np.array(layer["W"]), "b": np.array(layer["b"])}
            for layer in disc_data
        ]
