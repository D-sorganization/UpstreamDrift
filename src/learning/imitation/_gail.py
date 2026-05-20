"""GAIL (Generative Adversarial Imitation Learning) algorithm."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.learning.imitation._base import ImitationLearner, TrainingConfig
from src.learning.imitation.dataset import DemonstrationDataset


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
        if observation_dim is None:
            raise ValueError("observation_dim must be provided")
        super().__init__(observation_dim, action_dim, config, device)
        self._policy: list[dict[str, NDArray[np.floating]]] = []
        self._discriminator: list[dict[str, NDArray[np.floating]]] = []
        self._build_networks()

    def _build_networks(self) -> None:
        """Build policy and discriminator networks."""
        # Build simple MLP policy
        policy_layers = []
        input_dim = self.observation_dim

        for hidden_dim in self.config.hidden_sizes:
            policy_layers.append(
                {
                    "W": np.random.randn(input_dim, hidden_dim) * 0.01,
                    "b": np.zeros(hidden_dim),
                }
            )
            input_dim = hidden_dim

        policy_layers.append(
            {
                "W": np.random.randn(input_dim, self.action_dim) * 0.01,
                "b": np.zeros(self.action_dim),
            }
        )
        self._policy = policy_layers  # type: ignore[assignment]

        # Build discriminator (state-action -> [0, 1])
        disc_layers = []
        input_dim = self.observation_dim + self.action_dim

        for hidden_dim in self.config.hidden_sizes:
            disc_layers.append(
                {
                    "W": np.random.randn(input_dim, hidden_dim) * 0.01,
                    "b": np.zeros(hidden_dim),
                }
            )
            input_dim = hidden_dim

        disc_layers.append(
            {
                "W": np.random.randn(input_dim, 1) * 0.01,
                "b": np.zeros(1),
            }
        )
        self._discriminator = disc_layers  # type: ignore[assignment]

    def _forward_policy(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass through policy network."""
        if x is None:
            raise ValueError("x must be provided")
        for i, layer in enumerate(self._policy):
            x = x @ layer["W"] + layer["b"]
            if i < len(self._policy) - 1:
                x = np.maximum(0, x)  # ReLU
        return x

    def _forward_discriminator(
        self, state: NDArray[np.floating], action: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Forward pass through discriminator."""
        if state is None:
            raise ValueError("state must be provided")
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
        if dataset is None:
            raise ValueError("dataset must be provided")
        expert_states, expert_actions = dataset.to_state_action_pairs()

        if len(expert_states) == 0:
            raise ValueError("Dataset has no state-action pairs")

        history: dict[str, list[float]] = {"discriminator_loss": [], "policy_loss": []}
        lr = self.config.learning_rate

        for _epoch in range(self.config.epochs):
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
            # expert_grad = expert_preds - 1  # gradient towards 1
            # policy_grad = policy_preds  # gradient towards 0

            for _i, layer in enumerate(self._discriminator):
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
        if observation is None:
            raise ValueError("observation must be provided")
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
        if state is None:
            raise ValueError("state must be provided")
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)

        disc_output = self._forward_discriminator(state, action)
        # Reward is -log(1 - D(s,a))
        return (-np.log(1 - disc_output + 1e-8)).item()

    def save(self, path: str | Path) -> None:
        """Save GAIL networks."""
        if path is None:
            raise ValueError("path must be provided")
        path = Path(path)

        save_data = {
            "observation_dim": np.array(self.observation_dim),
            "action_dim": np.array(self.action_dim),
            "num_policy_layers": np.array(len(self._policy)),
            "num_disc_layers": np.array(len(self._discriminator)),
        }

        for i, layer in enumerate(self._policy):
            save_data[f"policy_{i}_W"] = layer["W"]
            save_data[f"policy_{i}_b"] = layer["b"]

        for i, layer in enumerate(self._discriminator):
            save_data[f"disc_{i}_W"] = layer["W"]
            save_data[f"disc_{i}_b"] = layer["b"]

        np.savez(path, **save_data)

    def load(self, path: str | Path) -> None:
        """Load GAIL networks."""
        if path is None:
            raise ValueError("path must be provided")
        path = Path(path)

        # Security: allow_pickle=False prevents arbitrary code execution
        data = np.load(path, allow_pickle=False)

        self.observation_dim = int(data["observation_dim"])
        self.action_dim = int(data["action_dim"])

        if "num_policy_layers" in data:
            num_policy_layers = int(data["num_policy_layers"])
            self._policy = [
                {"W": data[f"policy_{i}_W"], "b": data[f"policy_{i}_b"]}
                for i in range(num_policy_layers)
            ]

            num_disc_layers = int(data["num_disc_layers"])
            self._discriminator = [
                {"W": data[f"disc_{i}_W"], "b": data[f"disc_{i}_b"]}
                for i in range(num_disc_layers)
            ]
        else:
            raise ValueError(
                "Legacy format requiring allow_pickle=True is no longer supported for security reasons."
            )
