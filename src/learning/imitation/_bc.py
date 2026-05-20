"""Behavior Cloning imitation learning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import json
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.learning.imitation._base import ImitationLearner, TrainingConfig
from src.learning.imitation.dataset import DemonstrationDataset


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
        if observation_dim is None:
            raise ValueError("observation_dim must be provided")
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
            layers.append(
                {
                    "W": np.random.randn(input_dim, hidden_dim) * 0.01,
                    "b": np.zeros(hidden_dim),
                }
            )
            input_dim = hidden_dim

        # Output layer
        layers.append(
            {
                "W": np.random.randn(input_dim, self.action_dim) * 0.01,
                "b": np.zeros(self.action_dim),
            }
        )

        self._policy = layers

    def _forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass through network.

        Args:
            x: Input observations.

        Returns:
            Predicted actions.
        """
        if x is None:
            raise ValueError("x must be provided")
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
        if observations is None:
            raise ValueError("observations must be provided")
        predictions = self._forward(observations)
        diff = predictions - actions
        # ⚡ Bolt: np.vdot is ~2x faster than np.mean(diff**2) and avoids temporary array allocations
        return float(np.vdot(diff, diff) / diff.size)

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
        if observations is None:
            raise ValueError("observations must be provided")
        batch_size = len(observations)

        # Forward pass with caching
        activations = [observations]
        x = observations
        for i, layer in enumerate(self._policy):
            z = x @ layer["W"] + layer["b"]
            x = np.maximum(0, z) if i < len(self._policy) - 1 else z  # Linear output
            activations.append(x)

        # Backward pass
        gradients: list[dict[str, NDArray[np.floating]]] = []
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
        if dataset is None:
            raise ValueError("dataset must be provided")
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
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        lr = self.config.learning_rate

        for _epoch in range(self.config.epochs):
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
                for layer, grad in zip(self._policy, gradients, strict=True):
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
        if observation is None:
            raise ValueError("observation must be provided")
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
        if path is None:
            raise ValueError("path must be provided")
        path = Path(path)

        save_data = {
            "observation_dim": np.array(self.observation_dim),
            "action_dim": np.array(self.action_dim),
            "num_layers": np.array(len(self._policy)),
        }

        config_dict = {
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "hidden_sizes": self.config.hidden_sizes,
        }
        save_data["config_json"] = np.array(json.dumps(config_dict))

        for i, layer in enumerate(self._policy):
            save_data[f"layer_{i}_W"] = layer["W"]
            save_data[f"layer_{i}_b"] = layer["b"]

        np.savez(path, **save_data)

    def load(self, path: str | Path) -> None:
        """Load policy from disk.

        Args:
            path: Path to load file.
        """
        if path is None:
            raise ValueError("path must be provided")
        path = Path(path)

        # Security: allow_pickle=False prevents arbitrary code execution
        data = np.load(path, allow_pickle=False)

        self.observation_dim = int(data["observation_dim"])
        self.action_dim = int(data["action_dim"])

        if "num_layers" in data:
            num_layers = int(data["num_layers"])
            self._policy = [
                {"W": data[f"layer_{i}_W"], "b": data[f"layer_{i}_b"]}
                for i in range(num_layers)
            ]
        else:
            raise ValueError(
                "Legacy format requiring allow_pickle=True is no longer supported for security reasons."
            )
