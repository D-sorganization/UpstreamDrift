"""Checkpoint system for state saving and restoration.

This module provides the core components for the checkpoint/restore system,
enabling reversibility, debugging, and scenario exploration.

It defines:
- StateCheckpoint: A data container for simulation state.
- Checkpointable: A protocol for objects that can be checkpointed.
- CheckpointManager: A manager for handling checkpoint history and serialization.
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

from src.shared.python.ball_flight_physics import (
    EnvironmentalConditions,
    TrajectoryPoint,
)
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# Type aliases for state components
BallState: TypeAlias = TrajectoryPoint
EnvironmentState: TypeAlias = EnvironmentalConditions


@dataclass(frozen=True)
class StateCheckpoint:
    """Immutable container for a complete system state snapshot.

    Attributes:
        timestamp: Simulation time of the checkpoint.
        engine_state: State of the physics engine (q, v, etc.).
        ball_state: State of the ball (position, velocity, etc.).
        environment: State of the environment (wind, gravity, etc.).
    """

    timestamp: float
    engine_state: dict[str, Any] | None = None
    ball_state: BallState | None = None
    environment: EnvironmentState | None = None


@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for objects that support state checkpointing and restoration."""

    def save_checkpoint(self) -> StateCheckpoint:
        """Create a snapshot of the current state.

        Returns:
            A StateCheckpoint object containing the current state.
        """
        ...

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore state from a checkpoint.

        Args:
            checkpoint: The checkpoint to restore from.
        """
        ...


class CheckpointManager:
    """Manages creation, storage, and restoration of checkpoints.

    Features:
    - Circular buffer for memory-efficient history.
    - Time-based lookup.
    - Disk serialization/deserialization.
    """

    def __init__(self, target: Checkpointable, max_history: int = 100):
        """Initialize the checkpoint manager.

        Args:
            target: The object to manage checkpoints for.
            max_history: Maximum number of checkpoints to keep in memory.
        """
        self.target = target
        self._history: deque[StateCheckpoint] = deque(maxlen=max_history)

    def save_checkpoint(self) -> StateCheckpoint:
        """Capture current state and add to history.

        Returns:
            The created checkpoint.
        """
        checkpoint = self.target.save_checkpoint()
        self._history.append(checkpoint)
        logger.debug(f"Saved checkpoint at t={checkpoint.timestamp:.4f}")
        return checkpoint

    def restore_checkpoint(self, index: int = -1) -> None:
        """Restore from a checkpoint in history by index.

        Args:
            index: Index of the checkpoint (default: -1, most recent).

        Raises:
            IndexError: If history is empty or index is out of range.
        """
        if not self._history:
            raise IndexError("No checkpoints available to restore.")

        checkpoint = self._history[index]
        self.target.restore_checkpoint(checkpoint)
        logger.info(f"Restored checkpoint from t={checkpoint.timestamp:.4f}")

    def restore_by_timestamp(self, timestamp: float, tolerance: float = 1e-4) -> None:
        """Find and restore a checkpoint closest to the given timestamp.

        Args:
            timestamp: Target simulation time.
            tolerance: Maximum allowed time difference.

        Raises:
            ValueError: If no matching checkpoint is found.
        """
        if not self._history:
            raise ValueError("No checkpoints available.")

        # Find closest checkpoint
        closest = min(self._history, key=lambda cp: abs(cp.timestamp - timestamp))

        if abs(closest.timestamp - timestamp) > tolerance:
            raise ValueError(
                f"No checkpoint found near t={timestamp} (closest: {closest.timestamp})"
            )

        self.target.restore_checkpoint(closest)
        logger.info(
            f"Restored checkpoint for t={timestamp} (found t={closest.timestamp})"
        )

    def serialize(self, path: str | Path) -> None:
        """Save checkpoint history to disk.

        Args:
            path: File path to save to.
        """
        try:
            with open(path, "wb") as f:
                pickle.dump(list(self._history), f)
            logger.info(f"Serialized {len(self._history)} checkpoints to {path}")
        except Exception as e:
            logger.error(f"Failed to serialize checkpoints: {e}")
            raise

    def deserialize(self, path: str | Path) -> None:
        """Load checkpoint history from disk.

        Args:
            path: File path to load from.
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._history = deque(data, maxlen=self._history.maxlen)
            logger.info(f"Deserialized {len(self._history)} checkpoints from {path}")
        except Exception as e:
            logger.error(f"Failed to deserialize checkpoints: {e}")
            raise

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()

    @property
    def history_count(self) -> int:
        """Return number of stored checkpoints."""
        return len(self._history)
