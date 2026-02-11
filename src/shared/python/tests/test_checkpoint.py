import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.shared.python.base_physics_engine import BasePhysicsEngine, EngineState
from src.shared.python.checkpoint import (
    Checkpointable,
    CheckpointManager,
    StateCheckpoint,
)


class TestStateCheckpoint(unittest.TestCase):
    def test_creation(self) -> None:
        cp = StateCheckpoint(timestamp=1.0)
        self.assertEqual(cp.timestamp, 1.0)
        self.assertIsNone(cp.engine_state)
        self.assertIsNone(cp.ball_state)
        self.assertIsNone(cp.environment)

        cp2 = StateCheckpoint(
            timestamp=2.0,
            engine_state={"q": [1]},
            ball_state=MagicMock(),
            environment=MagicMock(),
        )
        self.assertEqual(cp2.timestamp, 2.0)
        self.assertEqual(cp2.engine_state, {"q": [1]})
        self.assertIsNotNone(cp2.ball_state)
        self.assertIsNotNone(cp2.environment)


class MockCheckpointable(Checkpointable):
    def __init__(self) -> None:
        self.current_state = 0
        self.time = 0.0

    def save_checkpoint(self) -> StateCheckpoint:
        return StateCheckpoint(
            timestamp=self.time, engine_state={"val": self.current_state}
        )

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        self.time = checkpoint.timestamp
        if checkpoint.engine_state:
            self.current_state = checkpoint.engine_state["val"]


class TestCheckpointManager(unittest.TestCase):
    def setUp(self) -> None:
        self.target = MockCheckpointable()
        self.manager = CheckpointManager(self.target, max_history=5)

    def test_save_and_restore(self) -> None:
        # Initial state
        self.target.current_state = 10
        self.target.time = 1.0
        self.manager.save_checkpoint()

        # Change state
        self.target.current_state = 20
        self.target.time = 2.0
        self.manager.save_checkpoint()

        # Change state again
        self.target.current_state = 30
        self.target.time = 3.0

        # Restore last checkpoint
        self.manager.restore_checkpoint()
        self.assertEqual(self.target.current_state, 20)
        self.assertEqual(self.target.time, 2.0)

        # Restore first checkpoint (index 0)
        self.manager.restore_checkpoint(0)
        self.assertEqual(self.target.current_state, 10)
        self.assertEqual(self.target.time, 1.0)

    def test_circular_buffer(self) -> None:
        for i in range(10):
            self.target.current_state = i
            self.target.time = float(i)
            self.manager.save_checkpoint()

        self.assertEqual(self.manager.history_count, 5)
        # Oldest should be i=5
        self.assertEqual(self.manager._history[0].engine_state["val"], 5)
        # Newest should be i=9
        self.assertEqual(self.manager._history[-1].engine_state["val"], 9)

    def test_restore_by_timestamp(self) -> None:
        self.target.time = 1.0
        self.manager.save_checkpoint()
        self.target.time = 2.0
        self.manager.save_checkpoint()

        self.manager.restore_by_timestamp(1.05, tolerance=0.1)
        self.assertEqual(self.target.time, 1.0)

        with self.assertRaises(ValueError):
            self.manager.restore_by_timestamp(3.0)


class TestBasePhysicsEngineCheckpoint(unittest.TestCase):
    def setUp(self) -> None:
        # Patch abstract methods so we can instantiate BasePhysicsEngine
        # We need to list all abstract methods
        patcher = patch.multiple(BasePhysicsEngine, __abstractmethods__=set())
        patcher.start()
        self.addCleanup(patcher.stop)

        self.engine = BasePhysicsEngine()
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
            timestamp=2.0,
            engine_state={
                "q": np.array([3.0, 4.0]),
                "v": np.array([0.3, 0.4]),
                "t": 2.0,
            },
        )
        self.engine.restore_checkpoint(cp)

        self.assertEqual(self.engine.state.time, 2.0)
        np.testing.assert_array_equal(self.engine.state.q, np.array([3.0, 4.0]))
        np.testing.assert_array_equal(self.engine.state.v, np.array([0.3, 0.4]))

    def test_restore_checkpoint_partial(self) -> None:
        # Should handle missing engine_state gracefully
        original_time = self.engine.state.time
        cp = StateCheckpoint(timestamp=3.0, engine_state=None)
        self.engine.restore_checkpoint(cp)
        self.assertEqual(self.engine.state.time, original_time)
