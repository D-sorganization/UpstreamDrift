"""Tests for src.engines.common.state module."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.common.state import (
    EngineLifecycleState,
    EngineStateMixin,
    SimulationState,
    StateManager,
)


class TestEngineLifecycleState:
    """Tests for EngineLifecycleState enum."""

    def test_all_states_exist(self) -> None:
        assert EngineLifecycleState.UNINITIALIZED is not None
        assert EngineLifecycleState.LOADING is not None
        assert EngineLifecycleState.INITIALIZED is not None
        assert EngineLifecycleState.STEPPING is not None
        assert EngineLifecycleState.PAUSED is not None
        assert EngineLifecycleState.ERROR is not None


class TestSimulationState:
    """Tests for SimulationState dataclass."""

    def test_default_construction(self) -> None:
        state = SimulationState(
            q=np.zeros(3),
            v=np.zeros(3),
        )
        assert state.time == 0.0
        assert state.step_count == 0
        assert state.metadata == {}

    def test_copy_is_deep(self) -> None:
        state = SimulationState(
            q=np.array([1.0, 2.0]),
            v=np.array([3.0, 4.0]),
            time=1.0,
        )
        copy = state.copy()
        copy.q[0] = 999.0
        assert state.q[0] == 1.0  # original unchanged

    def test_validate_valid_state(self) -> None:
        state = SimulationState(
            q=np.array([1.0, 2.0]),
            v=np.array([3.0, 4.0]),
        )
        assert state.validate() is True

    def test_validate_nan_q(self) -> None:
        state = SimulationState(
            q=np.array([float("nan")]),
            v=np.array([0.0]),
        )
        assert state.validate() is False


class TestStateManager:
    """Tests for StateManager class."""

    def test_construction(self) -> None:
        sm = StateManager(nq=3, nv=3)
        assert sm is not None
        assert sm.lifecycle == EngineLifecycleState.UNINITIALIZED

    def test_initialize(self) -> None:
        sm = StateManager(nq=3, nv=3)
        sm.initialize()
        assert sm.lifecycle == EngineLifecycleState.INITIALIZED

    def test_initialize_with_q0(self) -> None:
        q0 = np.array([1.0, 2.0, 3.0])
        sm = StateManager(nq=3, nv=3)
        sm.initialize(q0=q0)
        q, v = sm.get_state()
        np.testing.assert_array_equal(q, q0)
        np.testing.assert_array_equal(v, np.zeros(3))

    def test_set_state(self) -> None:
        sm = StateManager(nq=2, nv=2)
        sm.initialize()
        sm.set_state(np.array([5.0, 6.0]), np.array([7.0, 8.0]))
        q, v = sm.get_state()
        assert q[0] == pytest.approx(5.0)
        assert v[1] == pytest.approx(8.0)

    def test_set_state_wrong_dim(self) -> None:
        sm = StateManager(nq=2, nv=2)
        sm.initialize()
        with pytest.raises(ValueError):
            sm.set_state(np.array([1.0]), np.array([2.0, 3.0]))

    def test_advance_time(self) -> None:
        sm = StateManager(nq=1, nv=1)
        sm.initialize()
        sm.advance_time(0.01)
        assert sm.state.time == pytest.approx(0.01)
        sm.advance_time(0.01)
        assert sm.state.time == pytest.approx(0.02)

    def test_reset(self) -> None:
        sm = StateManager(nq=2, nv=2)
        sm.initialize(q0=np.array([1.0, 2.0]))
        sm.advance_time(1.0)
        sm.set_state(np.array([99.0, 99.0]), np.array([88.0, 88.0]))
        sm.reset()
        q, v = sm.get_state()
        # reset() zeros everything (no initial state memory)
        np.testing.assert_array_equal(q, [0.0, 0.0])
        np.testing.assert_array_equal(v, [0.0, 0.0])
        assert sm.state.time == pytest.approx(0.0)

    def test_undo_after_init(self) -> None:
        """set_state doesn't push history, so only initialize() is in buffer.
        After init, history has 1 entry (index=0), can_undo() returns False.
        """
        sm = StateManager(nq=1, nv=1)
        sm.initialize()
        # Only 1 entry in history from initialize(), no undo available
        assert not sm.can_undo()

    def test_undo_empty(self) -> None:
        sm = StateManager(nq=1, nv=1)
        sm.initialize()
        assert not sm.can_undo()
        assert not sm.undo()

    def test_redo_empty(self) -> None:
        sm = StateManager(nq=1, nv=1)
        sm.initialize()
        assert not sm.can_redo()
        assert not sm.redo()


class TestEngineStateMixin:
    """Tests for EngineStateMixin."""

    def test_mixin_init(self) -> None:
        mixin = EngineStateMixin()
        mixin.__init__()
        assert mixin._get_lifecycle() == EngineLifecycleState.UNINITIALIZED

    def test_lifecycle_callback(self) -> None:
        mixin = EngineStateMixin()
        mixin.__init__()
        states_seen: list[EngineLifecycleState] = []
        mixin.add_lifecycle_callback(lambda s: states_seen.append(s))
        mixin._set_lifecycle(EngineLifecycleState.INITIALIZED)
        assert EngineLifecycleState.INITIALIZED in states_seen

    def test_remove_lifecycle_callback(self) -> None:
        mixin = EngineStateMixin()
        mixin.__init__()
        states_seen: list[EngineLifecycleState] = []
        cb = lambda s: states_seen.append(s)  # noqa: E731
        mixin.add_lifecycle_callback(cb)
        mixin.remove_lifecycle_callback(cb)
        mixin._set_lifecycle(EngineLifecycleState.INITIALIZED)
        assert len(states_seen) == 0
