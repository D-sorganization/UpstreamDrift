"""Behavior Cloning imitation learning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from src.learning.imitation._base import (
    ImitationLearner,
    TrainingConfig,
    validated_state_action_pairs,
)
from src.learning.imitation._mlp import (
    _adam_step,
    _build_mlp,
    _init_adam_state,
    _mlp_backward,
    _mlp_forward,
)
from src.learning.imitation.dataset import DemonstrationDataset

logger = logging.getLogger(__name__)


def _warn_if_loss_stalled(train_loss: list[float]) -> None:
    """Postcondition check: warn when training did not cut the loss by 1 %."""
    if len(train_loss) < 2:
        return
    initial_loss, final_loss = train_loss[0], train_loss[-1]
    if initial_loss > 0 and final_loss >= initial_loss * 0.99:
        msg = (
            f"Training did not decrease loss significantly: "
            f"initial loss {initial_loss:.4f}, final loss {final_loss:.4f}."
        )
        logger.warning(msg)
        # stacklevel=3 points at the caller of train(), as before the split.
        warnings.warn(msg, UserWarning, stacklevel=3)


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
        """Build the neural network policy with He/Xavier initialisation."""
        self._policy = _build_mlp(
            input_dim=self.observation_dim,
            output_dim=self.action_dim,
            hidden_sizes=self.config.hidden_sizes,
            init="he",
        )

    def _forward(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Forward pass through network.

        Args:
            x: Input observations.

        Returns:
            Predicted actions.
        """
        if x is None:
            raise ValueError("x must be provided")
        out, _ = _mlp_forward(self._policy, x)
        return out

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
        if actions is None:
            raise ValueError("actions must be provided")
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
        if actions is None:
            raise ValueError("actions must be provided")
        batch_size = len(observations)
        predictions, activations = _mlp_forward(self._policy, observations)
        delta = 2.0 * (predictions - actions) / batch_size
        gradients, _ = _mlp_backward(self._policy, activations, delta)
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

        Raises:
            ValueError: If dataset is invalid or validation_split is out of bounds.
        """
        observations, actions = validated_state_action_pairs(dataset, validation_split)

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

        if len(train_obs) == 0:
            raise ValueError("validation_split leaves no training samples")

        # Training loop
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        batch_size = max(1, min(self.config.batch_size, len(train_obs)))
        adam_state = _init_adam_state(self._policy)
        step = 0

        for _epoch in range(self.config.epochs):
            # Shuffle training data
            perm = np.random.permutation(len(train_obs))
            train_obs = train_obs[perm]
            train_act = train_act[perm]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_obs), batch_size):
                batch_obs = train_obs[i : i + batch_size]
                batch_act = train_act[i : i + batch_size]
                step += 1

                # Compute gradients
                gradients = self._backward(batch_obs, batch_act)

                # Update weights via Adam
                _adam_step(
                    self._policy,
                    gradients,
                    adam_state,
                    lr,
                    step,
                    weight_decay,
                )

                epoch_loss += self._compute_loss(batch_obs, batch_act)
                n_batches += 1

            # Record metrics
            train_loss = epoch_loss / n_batches
            val_loss = self._compute_loss(val_obs, val_act)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

        _warn_if_loss_stalled(history["train_loss"])
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

        np.savez(path, **save_data)  # type: ignore[arg-type]

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
