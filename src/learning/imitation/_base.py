"""Base classes for imitation learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    spaces = None  # type: ignore[assignment]

from src.learning.imitation.dataset import DemonstrationDataset


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
        if observation_dim is None:
            raise ValueError("observation_dim must be provided")
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
    ) -> ImitationLearner:
        """Create learner from Gymnasium spaces.

        Args:
            observation_space: Observation space.
            action_space: Action space.
            config: Training configuration.
            device: Compute device.

        Returns:
            Imitation learner instance.
        """
        if observation_space is None:
            raise ValueError("observation_space must be provided")
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

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save trained policy.

        Args:
            path: Path to save file.
        """

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load trained policy.

        Args:
            path: Path to load file.
        """

    def get_training_history(self) -> dict[str, list[float]]:
        """Get training history.

        Returns:
            Dictionary with training metrics over epochs.
        """
        return self._training_history.copy()
