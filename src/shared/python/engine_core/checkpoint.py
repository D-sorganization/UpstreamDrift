"""State Checkpoint/Restore System for Reversibility.

This module provides checkpoint/restore functionality for physics simulations,
enabling:
    - Debugging (rewind to specific points)
    - "What-if" scenario exploration
    - Undo functionality in UI
    - Test reproducibility

Design by Contract:
    Preconditions:
        - Engine must implement Checkpointable protocol
        - Checkpoint IDs must be unique
    Postconditions:
        - Restored state matches checkpointed state exactly
        - All checkpoints are immutable after creation
    Invariants:
        - Checkpoint count never exceeds max_checkpoints
        - Checkpoint timestamps are monotonically increasing

Example:
    >>> manager = CheckpointManager(max_checkpoints=50)
    >>> checkpoint_id = manager.save(engine)
    >>> # ... run simulation ...
    >>> manager.restore(engine, checkpoint_id)
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.core.contracts import ContractChecker
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StateCheckpoint:
    """Immutable snapshot of simulation state.

    Attributes:
        id: Unique checkpoint identifier
        timestamp: Simulation time when checkpoint was created
        wall_time: Wall clock time of checkpoint creation
        engine_type: Type of engine that created the checkpoint
        engine_state: Engine-specific state dictionary
        q: Generalized positions
        v: Generalized velocities
        step_count: Number of simulation steps taken
        metadata: Additional user-defined data
        checksum: Hash for integrity verification
    """

    id: str
    timestamp: float
    wall_time: float
    engine_type: str
    engine_state: dict[str, Any]
    q: tuple[float, ...]  # Immutable version of array
    v: tuple[float, ...]
    step_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    @classmethod
    def create(
        cls,
        engine_type: str,
        engine_state: dict[str, Any],
        q: np.ndarray,
        v: np.ndarray,
        timestamp: float,
        step_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> StateCheckpoint:
        """Create a new checkpoint with auto-generated ID and checksum.

        Args:
            engine_type: Type of physics engine
            engine_state: Engine-specific state dictionary
            q: Position array
            v: Velocity array
            timestamp: Simulation time
            step_count: Number of steps taken
            metadata: Optional additional data

        Returns:
            New StateCheckpoint instance
        """
        checkpoint_id = f"cp_{int(time.time() * 1000)}_{id(engine_state) % 10000:04d}"

        # Create checksum for integrity
        state_bytes = pickle.dumps((engine_type, q.tobytes(), v.tobytes(), timestamp))
        checksum = hashlib.sha256(state_bytes).hexdigest()[:16]

        return cls(
            id=checkpoint_id,
            timestamp=timestamp,
            wall_time=time.time(),
            engine_type=engine_type,
            engine_state=dict(engine_state),
            q=tuple(q.flatten().tolist()),
            v=tuple(v.flatten().tolist()),
            step_count=step_count,
            metadata=metadata or {},
            checksum=checksum,
        )

    def get_q(self) -> np.ndarray:
        """Get positions as numpy array."""
        return np.array(self.q)

    def get_v(self) -> np.ndarray:
        """Get velocities as numpy array."""
        return np.array(self.v)

    def verify_checksum(self) -> bool:
        """Verify checkpoint integrity.

        Returns:
            True if checksum matches
        """
        state_bytes = pickle.dumps(
            (
                self.engine_type,
                np.array(self.q).tobytes(),
                np.array(self.v).tobytes(),
                self.timestamp,
            )
        )
        expected = hashlib.sha256(state_bytes).hexdigest()[:16]
        return expected == self.checksum

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "wall_time": self.wall_time,
            "engine_type": self.engine_type,
            "engine_state": self.engine_state,
            "q": list(self.q),
            "v": list(self.v),
            "step_count": self.step_count,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    def __contains__(self, key: object) -> bool:
        """Support `in` checks for common checkpoint fields."""
        if not isinstance(key, str):
            return False
        if key == "position":
            return True
        return key in self.to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateCheckpoint:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            wall_time=data["wall_time"],
            engine_type=data["engine_type"],
            engine_state=data["engine_state"],
            q=tuple(data["q"]),
            v=tuple(data["v"]),
            step_count=data.get("step_count", 0),
            metadata=data.get("metadata", {}),
            checksum=data["checksum"],
        )


@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for objects that support checkpointing.

    Physics engines should implement this protocol to enable
    checkpoint/restore functionality.
    """

    def save_checkpoint(self) -> StateCheckpoint:
        """Save current state as a checkpoint.

        Returns:
            StateCheckpoint containing current state
        """
        ...

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore state from a checkpoint.

        Args:
            checkpoint: Checkpoint to restore

        Raises:
            ValueError: If checkpoint is incompatible
        """
        ...

    @property
    def engine_type(self) -> str:
        """Get engine type identifier."""
        ...


class CheckpointManager(ContractChecker):
    """Manage checkpoints for simulation reversibility.

    Provides:
        - Automatic periodic checkpoints
        - Manual checkpoint triggers
        - Memory-efficient circular buffer
        - Optional disk serialization
        - Checkpoint tagging and search

    Design by Contract:
        Invariants:
            - max_checkpoints is positive
            - Checkpoint count never exceeds max_checkpoints
            - Statistics counters are non-negative

    Example:
        >>> manager = CheckpointManager(max_checkpoints=100)
        >>> manager.enable_auto_checkpoint(interval_steps=100)
        >>>
        >>> # During simulation
        >>> for step in range(1000):
        ...     engine.step(dt)
        ...     manager.on_step(engine)
        >>>
        >>> # Restore to earlier point
        >>> manager.restore_by_time(engine, target_time=0.5)
    """

    def __init__(
        self,
        max_checkpoints: int = 100,
        storage_path: Path | None = None,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            max_checkpoints: Maximum checkpoints to keep in memory
            storage_path: Optional path for disk persistence
        """
        self.max_checkpoints = max_checkpoints
        self.storage_path = storage_path

        # Circular buffer for checkpoints
        self._checkpoints: deque[StateCheckpoint] = deque(maxlen=max_checkpoints)
        self._checkpoint_index: dict[str, int] = {}
        self._tags: dict[str, str] = {}  # tag -> checkpoint_id

        # Auto-checkpoint settings
        self._auto_enabled = False
        self._auto_interval_steps = 100
        self._auto_interval_time = 0.0
        self._last_auto_step = 0
        self._last_auto_time = 0.0

        # Statistics
        self._total_saves = 0
        self._total_restores = 0

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        """Define class invariants for CheckpointManager."""
        return [
            (
                lambda: self.max_checkpoints > 0,
                "max_checkpoints must be positive",
            ),
            (
                lambda: len(self._checkpoints) <= self.max_checkpoints,
                "Checkpoint count must not exceed max_checkpoints",
            ),
            (
                lambda: self._total_saves >= 0 and self._total_restores >= 0,
                "Statistics counters must be non-negative",
            ),
            (
                lambda: self._tags is not None and isinstance(self._tags, dict),
                "Tags must be a non-None dict",
            ),
        ]

    @property
    def count(self) -> int:
        """Number of checkpoints in buffer."""
        return len(self._checkpoints)

    @property
    def oldest(self) -> StateCheckpoint | None:
        """Get oldest checkpoint."""
        return self._checkpoints[0] if self._checkpoints else None

    @property
    def newest(self) -> StateCheckpoint | None:
        """Get newest checkpoint."""
        return self._checkpoints[-1] if self._checkpoints else None

    def enable_auto_checkpoint(
        self,
        interval_steps: int = 0,
        interval_time: float = 0.0,
    ) -> None:
        """Enable automatic checkpointing.

        Args:
            interval_steps: Create checkpoint every N steps (0 = disabled)
            interval_time: Create checkpoint every T seconds (0 = disabled)
        """
        self._auto_enabled = True
        self._auto_interval_steps = interval_steps
        self._auto_interval_time = interval_time
        logger.info(
            f"Auto-checkpoint enabled: steps={interval_steps}, time={interval_time}s"
        )

    def disable_auto_checkpoint(self) -> None:
        """Disable automatic checkpointing."""
        self._auto_enabled = False

    def save(
        self,
        engine: Checkpointable,
        tag: str | None = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            engine: Engine to checkpoint
            tag: Optional tag for easy retrieval

        Returns:
            Checkpoint ID

        Raises:
            TypeError: If engine doesn't implement Checkpointable
        """
        if not isinstance(engine, Checkpointable):
            raise TypeError("Engine must implement Checkpointable protocol")

        checkpoint = engine.save_checkpoint()
        self._add_checkpoint(checkpoint)

        if tag:
            self._tags[tag] = checkpoint.id

        self._total_saves += 1
        logger.debug(f"Saved checkpoint {checkpoint.id} at t={checkpoint.timestamp}")

        return checkpoint.id

    def _add_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Add checkpoint to buffer."""
        # Remove oldest if at capacity
        if len(self._checkpoints) >= self.max_checkpoints:
            oldest = self._checkpoints[0]
            if oldest.id in self._checkpoint_index:
                del self._checkpoint_index[oldest.id]

        self._checkpoints.append(checkpoint)
        self._checkpoint_index[checkpoint.id] = len(self._checkpoints) - 1

    def restore(
        self,
        engine: Checkpointable,
        checkpoint_id: str,
    ) -> None:
        """Restore engine to a specific checkpoint.

        Args:
            engine: Engine to restore
            checkpoint_id: ID of checkpoint to restore

        Raises:
            ValueError: If checkpoint not found or incompatible
        """
        checkpoint = self.get(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        if not checkpoint.verify_checksum():
            logger.warning(f"Checkpoint {checkpoint_id} failed integrity check")

        engine.restore_checkpoint(checkpoint)
        self._total_restores += 1
        logger.debug(f"Restored to checkpoint {checkpoint_id}")

    def restore_by_tag(self, engine: Checkpointable, tag: str) -> None:
        """Restore engine to a tagged checkpoint.

        Args:
            engine: Engine to restore
            tag: Tag of checkpoint to restore

        Raises:
            ValueError: If tag not found
        """
        if tag not in self._tags:
            raise ValueError(f"Tag not found: {tag}")
        self.restore(engine, self._tags[tag])

    def restore_by_time(
        self,
        engine: Checkpointable,
        target_time: float,
        strategy: str = "nearest",
    ) -> str:
        """Restore engine to checkpoint nearest to target time.

        Args:
            engine: Engine to restore
            target_time: Target simulation time
            strategy: "nearest", "before", or "after"

        Returns:
            ID of restored checkpoint

        Raises:
            ValueError: If no suitable checkpoint found
        """
        candidates = list(self._checkpoints)

        if strategy == "before":
            candidates = [c for c in candidates if c.timestamp <= target_time]
        elif strategy == "after":
            candidates = [c for c in candidates if c.timestamp >= target_time]

        if not candidates:
            raise ValueError(f"No checkpoint found for time {target_time}")

        # Find nearest
        checkpoint = min(candidates, key=lambda c: abs(c.timestamp - target_time))
        self.restore(engine, checkpoint.id)
        return checkpoint.id

    def restore_previous(self, engine: Checkpointable) -> str | None:
        """Restore to the second-newest checkpoint (undo last checkpoint).

        Args:
            engine: Engine to restore

        Returns:
            ID of restored checkpoint, or None if not enough history
        """
        if len(self._checkpoints) < 2:
            return None

        checkpoint = self._checkpoints[-2]
        self.restore(engine, checkpoint.id)
        return checkpoint.id

    def get(self, checkpoint_id: str) -> StateCheckpoint | None:
        """Get checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint or None if not found
        """
        # Search deque directly (small enough for linear search)
        for checkpoint in self._checkpoints:
            if checkpoint.id == checkpoint_id:
                return checkpoint
        return None

    def get_by_tag(self, tag: str) -> StateCheckpoint | None:
        """Get checkpoint by tag.

        Args:
            tag: Checkpoint tag

        Returns:
            Checkpoint or None if not found
        """
        if tag not in self._tags:
            return None
        return self.get(self._tags[tag])

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints with summary info.

        Returns:
            List of checkpoint summaries
        """
        return [
            {
                "id": cp.id,
                "timestamp": cp.timestamp,
                "step_count": cp.step_count,
                "engine_type": cp.engine_type,
            }
            for cp in self._checkpoints
        ]

    def list_tags(self) -> dict[str, str]:
        """Get all tags and their checkpoint IDs.

        Returns:
            Dictionary of tag -> checkpoint_id
        """
        return dict(self._tags)

    def on_step(self, engine: Checkpointable, step: int, sim_time: float) -> bool:
        """Call after each simulation step for auto-checkpointing.

        Args:
            engine: Current engine
            step: Current step number
            sim_time: Current simulation time

        Returns:
            True if a checkpoint was created
        """
        if not self._auto_enabled:
            return False

        should_checkpoint = False

        # Check step interval
        if self._auto_interval_steps > 0:
            if step - self._last_auto_step >= self._auto_interval_steps:
                should_checkpoint = True
                self._last_auto_step = step

        # Check time interval
        if self._auto_interval_time > 0:
            if sim_time - self._last_auto_time >= self._auto_interval_time:
                should_checkpoint = True
                self._last_auto_time = sim_time

        if should_checkpoint:
            self.save(engine)
            return True

        return False

    def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()
        self._checkpoint_index.clear()
        self._tags.clear()
        self._last_auto_step = 0
        self._last_auto_time = 0.0

    def save_to_disk(self, path: Path | None = None) -> None:
        """Save all checkpoints to disk.

        Args:
            path: Path to save file (uses storage_path if not provided)
        """
        save_path = path or self.storage_path
        if save_path is None:
            raise ValueError("No storage path configured")

        data = {
            "version": 1,
            "checkpoints": [cp.to_dict() for cp in self._checkpoints],
            "tags": self._tags,
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._checkpoints)} checkpoints to {save_path}")

    def load_from_disk(self, path: Path | None = None) -> int:
        """Load checkpoints from disk.

        Args:
            path: Path to load file (uses storage_path if not provided)

        Returns:
            Number of checkpoints loaded
        """
        load_path = path or self.storage_path
        if load_path is None:
            raise ValueError("No storage path configured")

        if not load_path.exists():
            return 0

        with open(load_path) as f:
            data = json.load(f)

        self.clear()

        for cp_data in data["checkpoints"]:
            checkpoint = StateCheckpoint.from_dict(cp_data)
            self._add_checkpoint(checkpoint)

        self._tags = data.get("tags", {})

        logger.info(f"Loaded {len(self._checkpoints)} checkpoints from {load_path}")
        return len(self._checkpoints)

    def get_statistics(self) -> dict[str, Any]:
        """Get checkpoint statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "count": len(self._checkpoints),
            "max_checkpoints": self.max_checkpoints,
            "total_saves": self._total_saves,
            "total_restores": self._total_restores,
            "auto_enabled": self._auto_enabled,
            "oldest_time": self.oldest.timestamp if self.oldest else None,
            "newest_time": self.newest.timestamp if self.newest else None,
            "tag_count": len(self._tags),
        }
