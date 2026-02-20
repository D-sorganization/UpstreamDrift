"""Tests for the checkpoint save/restore functionality."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.shared.python.engine_core.base_physics_engine import (
    BasePhysicsEngine,
    EngineState,
)
from src.shared.python.engine_core.checkpoint import (
    CheckpointManager,
    StateCheckpoint,
)


class TestStateCheckpoint(unittest.TestCase):
    def test_creation(self) -> None:
        cp = StateCheckpoint(
            id="test1",
            timestamp=1.0,
            wall_time=0.0,
            engine_type="test",
            engine_state={},
            q=(),
            v=(),
        )
        self.assertEqual(cp.timestamp, 1.0)
        self.assertEqual(cp.engine_state, {})
        self.assertEqual(cp.q, ())
        self.assertEqual(cp.v, ())

        cp2 = StateCheckpoint(
            id="test2",
            timestamp=2.0,
            wall_time=0.0,
            engine_type="test",
            engine_state={"q": [1]},
            q=(),
            v=(),
        )
        self.assertEqual(cp2.timestamp, 2.0)
        self.assertEqual(cp2.engine_state, {"q": [1]})
        self.assertIsNotNone(cp2.engine_state)


class MockCheckpointable:
    """Minimal Checkpointable implementation for testing."""

    def __init__(self) -> None:
        """Initialize with zero state and time."""
        self.current_state = 0
        self.time = 0.0

    @property
    def engine_type(self) -> str:
        """Return the engine type identifier."""
        return "mock"

    def save_checkpoint(self) -> StateCheckpoint:
        """Save current state into a checkpoint."""
        return StateCheckpoint(
            id=f"cp_{self.time}",
            timestamp=self.time,
            wall_time=0.0,
            engine_type=self.engine_type,
            engine_state={"val": self.current_state},
            q=(),
            v=(),
        )

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore state from a checkpoint."""
        self.time = checkpoint.timestamp
        if checkpoint.engine_state:
            self.current_state = checkpoint.engine_state["val"]


class TestCheckpointManager(unittest.TestCase):
    """Tests for CheckpointManager save, restore, and history."""

    def setUp(self) -> None:
        """Create a fresh MockCheckpointable and CheckpointManager."""
        self.target = MockCheckpointable()
        self.manager = CheckpointManager(max_checkpoints=5)

    def test_save_and_restore(self) -> None:
        # Initial state
        self.target.current_state = 10
        self.target.time = 1.0
        self.manager.save(self.target)

        # Change state
        self.target.current_state = 20
        self.target.time = 2.0
        self.manager.save(self.target)

        # Change state again
        self.target.current_state = 30
        self.target.time = 3.0

        # Restore last checkpoint
        newest = self.manager.newest
        assert newest is not None
        self.manager.restore(self.target, newest.id)
        self.assertEqual(self.target.current_state, 20)
        self.assertEqual(self.target.time, 2.0)

        # Restore first checkpoint (oldest)
        oldest = self.manager.oldest
        assert oldest is not None
        self.manager.restore(self.target, oldest.id)
        self.assertEqual(self.target.current_state, 10)
        self.assertEqual(self.target.time, 1.0)

    def test_circular_buffer(self) -> None:
        for i in range(10):
            self.target.current_state = i
            self.target.time = float(i)
            self.manager.save(self.target)

        self.assertEqual(self.manager.count, 5)
        # Oldest should be i=5
        self.assertEqual(self.manager._checkpoints[0].engine_state["val"], 5)
        # Newest should be i=9
        self.assertEqual(self.manager._checkpoints[-1].engine_state["val"], 9)

    def test_restore_by_timestamp(self) -> None:
        self.target.time = 1.0
        self.manager.save(self.target)
        self.target.time = 2.0
        self.manager.save(self.target)

        self.manager.restore_by_time(self.target, 1.05, strategy="nearest")
        self.assertEqual(self.target.time, 1.0)

        with self.assertRaises(ValueError):
            self.manager.restore_by_time(self.target, 3.0, strategy="after")


class TestBasePhysicsEngineCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        # Patch abstract methods so we can instantiate BasePhysicsEngine
        # We need to list all abstract methods
        patcher = patch.multiple(BasePhysicsEngine, __abstractmethods__=set())
        patcher.start()
        self.addCleanup(patcher.stop)

        self.engine = BasePhysicsEngine()  # type: ignore[abstract]
        self.engine._is_initialized = True
        self.engine.model = MagicMock()
        self.engine.state = EngineState(nq=2, nv=2)
        self.engine.state.q = np.array([1.0, 2.0])
        self.engine.state.v = np.array([0.1, 0.2])
        self.engine.state.time = 1.5

    def test_save_checkpoint(self) -> None:
        cp = self.engine.save_checkpoint()
        self.assertEqual(cp.timestamp, 1.5)
        self.assertIsNotNone(cp.engine_state)
        np.testing.assert_array_equal(cp.engine_state["q"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(cp.engine_state["v"], np.array([0.1, 0.2]))
        self.assertEqual(cp.engine_state["t"], 1.5)

    def test_restore_checkpoint(self) -> None:
        cp = StateCheckpoint(
            id="restore_test",
            timestamp=2.0,
            wall_time=0.0,
            engine_type="test",
            engine_state={
                "q": np.array([3.0, 4.0]),
                "v": np.array([0.3, 0.4]),
                "t": 2.0,
            },
            q=(3.0, 4.0),
            v=(0.3, 0.4),
        )
        self.engine.restore_checkpoint(cp)

        assert self.engine.state is not None
        self.assertEqual(self.engine.state.time, 2.0)
        np.testing.assert_array_equal(self.engine.state.q, np.array([3.0, 4.0]))
        np.testing.assert_array_equal(self.engine.state.v, np.array([0.3, 0.4]))

    def test_restore_checkpoint_partial(self) -> None:
        # Should handle missing engine_state gracefully
        assert self.engine.state is not None
        original_time = self.engine.state.time
        cp = StateCheckpoint(
            id="partial_test",
            timestamp=3.0,
            wall_time=0.0,
            engine_type="test",
            engine_state={},
            q=(),
            v=(),
        )
        self.engine.restore_checkpoint(cp)
        self.assertEqual(self.engine.state.time, original_time)
