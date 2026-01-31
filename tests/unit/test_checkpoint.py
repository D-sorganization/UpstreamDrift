"""Tests for the checkpoint/restore system.

Tests the StateCheckpoint and CheckpointManager classes for
simulation reversibility.
"""

import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.checkpoint import (
    CheckpointManager,
    StateCheckpoint,
)


class MockEngine:
    """Mock engine implementing Checkpointable protocol."""

    def __init__(self, nq: int = 7, nv: int = 6) -> None:
        self.nq = nq
        self.nv = nv
        self.q = np.zeros(nq)
        self.v = np.zeros(nv)
        self.time = 0.0
        self.step_count = 0
        self._engine_type = "MockEngine"

    @property
    def engine_type(self) -> str:
        return self._engine_type

    def save_checkpoint(self) -> StateCheckpoint:
        return StateCheckpoint.create(
            engine_type=self.engine_type,
            engine_state={"nq": self.nq, "nv": self.nv},
            q=self.q,
            v=self.v,
            timestamp=self.time,
            step_count=self.step_count,
        )

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        self.q = checkpoint.get_q()
        self.v = checkpoint.get_v()
        self.time = checkpoint.timestamp
        self.step_count = checkpoint.step_count

    def step(self, dt: float = 0.01) -> None:
        """Simulate a step."""
        self.q += np.ones(self.nq) * dt
        self.v += np.ones(self.nv) * dt * 2
        self.time += dt
        self.step_count += 1


class TestStateCheckpoint:
    """Tests for StateCheckpoint dataclass."""

    def test_create_checkpoint(self) -> None:
        """Test checkpoint creation."""
        q = np.array([1.0, 2.0, 3.0])
        v = np.array([0.1, 0.2, 0.3])

        checkpoint = StateCheckpoint.create(
            engine_type="TestEngine",
            engine_state={"key": "value"},
            q=q,
            v=v,
            timestamp=1.5,
            step_count=100,
        )

        assert checkpoint.engine_type == "TestEngine"
        assert checkpoint.timestamp == 1.5
        assert checkpoint.step_count == 100
        assert len(checkpoint.checksum) > 0

    def test_checkpoint_immutable(self) -> None:
        """Test checkpoint is immutable (frozen dataclass)."""
        checkpoint = StateCheckpoint.create(
            engine_type="Test",
            engine_state={},
            q=np.zeros(3),
            v=np.zeros(3),
            timestamp=0.0,
        )

        with pytest.raises(FrozenInstanceError):
            checkpoint.timestamp = 1.0  # type: ignore

    def test_get_arrays(self) -> None:
        """Test getting q and v as numpy arrays."""
        q = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])

        checkpoint = StateCheckpoint.create(
            engine_type="Test",
            engine_state={},
            q=q,
            v=v,
            timestamp=0.0,
        )

        assert np.allclose(checkpoint.get_q(), q)
        assert np.allclose(checkpoint.get_v(), v)

    def test_verify_checksum(self) -> None:
        """Test checksum verification."""
        checkpoint = StateCheckpoint.create(
            engine_type="Test",
            engine_state={},
            q=np.array([1.0, 2.0]),
            v=np.array([3.0, 4.0]),
            timestamp=1.0,
        )

        assert checkpoint.verify_checksum()

    def test_serialization(self) -> None:
        """Test to_dict and from_dict."""
        original = StateCheckpoint.create(
            engine_type="Test",
            engine_state={"key": "value"},
            q=np.array([1.0, 2.0]),
            v=np.array([3.0, 4.0]),
            timestamp=1.5,
            step_count=50,
        )

        data = original.to_dict()
        restored = StateCheckpoint.from_dict(data)

        assert restored.id == original.id
        assert restored.timestamp == original.timestamp
        assert np.allclose(restored.get_q(), original.get_q())


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def manager(self) -> CheckpointManager:
        """Create checkpoint manager."""
        return CheckpointManager(max_checkpoints=10)

    @pytest.fixture
    def engine(self) -> MockEngine:
        """Create mock engine."""
        return MockEngine()

    def test_save_checkpoint(
        self, manager: CheckpointManager, engine: MockEngine
    ) -> None:
        """Test saving a checkpoint."""
        engine.q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        engine.time = 0.5

        checkpoint_id = manager.save(engine)

        assert manager.count == 1
        assert checkpoint_id in [cp["id"] for cp in manager.list_checkpoints()]

    def test_restore_checkpoint(
        self, manager: CheckpointManager, engine: MockEngine
    ) -> None:
        """Test restoring a checkpoint."""
        # Save initial state
        engine.q = np.ones(7) * 5.0
        engine.time = 1.0
        manager.save(engine)

        # Modify state
        engine.q = np.zeros(7)
        engine.time = 2.0

        # Restore (get last saved)
        last_cp = manager.list_checkpoints()[-1]["id"]
        manager.restore(engine, last_cp)

        assert np.allclose(engine.q, np.ones(7) * 5.0)
        assert engine.time == 1.0

    def test_tagged_checkpoint(
        self, manager: CheckpointManager, engine: MockEngine
    ) -> None:
        """Test saving and restoring with tags."""
        engine.time = 0.5
        manager.save(engine, tag="impact")

        engine.time = 1.0
        manager.save(engine, tag="follow_through")

        # Restore by tag
        manager.restore_by_tag(engine, "impact")
        assert engine.time == 0.5

    def test_restore_by_time(
        self, manager: CheckpointManager, engine: MockEngine
    ) -> None:
        """Test restoring to nearest time."""
        # Save checkpoint at t=1.0
        engine.time = 1.0
        manager.save(engine)

        # Modify engine
        engine.time = 5.0

        # Restore to time nearest 1.0
        manager.restore_by_time(engine, target_time=1.0)

        # Should be restored to 1.0
        assert engine.time == 1.0

    def test_max_checkpoints(
        self, manager: CheckpointManager, engine: MockEngine
    ) -> None:
        """Test circular buffer respects max size."""
        # Save more than max
        for i in range(15):
            engine.time = float(i)
            manager.save(engine)

        assert manager.count == 10  # Should be capped

    def test_auto_checkpoint(
        self, manager: CheckpointManager, engine: MockEngine
    ) -> None:
        """Test automatic checkpointing."""
        manager.enable_auto_checkpoint(interval_steps=5)

        created = 0
        for step in range(20):
            engine.step()
            if manager.on_step(engine, step, engine.time):
                created += 1

        assert created >= 3  # Should have created at least 3 auto checkpoints

    def test_disk_persistence(
        self, manager: CheckpointManager, engine: MockEngine
    ) -> None:
        """Test saving and loading from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoints.json"

            # Save some checkpoints
            engine.q = np.ones(7) * 3.0
            engine.time = 0.5
            manager.save(engine, tag="test")
            manager.save_to_disk(path)

            # Create new manager and load
            new_manager = CheckpointManager()
            count = new_manager.load_from_disk(path)

            assert count == 1
            assert "test" in new_manager.list_tags()

    def test_restore_previous(
        self, manager: CheckpointManager, engine: MockEngine
    ) -> None:
        """Test undo functionality."""
        engine.time = 0.0
        manager.save(engine)

        engine.time = 1.0
        manager.save(engine)

        engine.time = 2.0

        # Undo to previous checkpoint
        checkpoint_id = manager.restore_previous(engine)

        assert checkpoint_id is not None
        assert engine.time == 0.0  # Should be at first checkpoint

    def test_statistics(self, manager: CheckpointManager, engine: MockEngine) -> None:
        """Test statistics reporting."""
        manager.save(engine)
        manager.save(engine, tag="tagged")

        stats = manager.get_statistics()

        assert stats["count"] == 2
        assert stats["total_saves"] == 2
        assert stats["tag_count"] == 1
