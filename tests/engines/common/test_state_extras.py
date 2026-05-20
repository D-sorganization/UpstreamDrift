"""Coverage for state.py — EngineStateMixin, ForceAccumulator, edges."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.common.state import (
    EngineLifecycleState,
    EngineStateMixin,
    ForceAccumulator,
    ForceSource,
    SimulationState,
    StateManager,
)
from src.shared.python.core.contracts import StateError


# ---------------------------------------------------------------------------
# SimulationState
# ---------------------------------------------------------------------------


class TestSimulationStateValidate:
    def test_valid_state(self) -> None:
        s = SimulationState(q=np.zeros(3), v=np.zeros(3))
        assert s.validate() is True

    def test_nan_q_invalid(self) -> None:
        s = SimulationState(q=np.array([np.nan, 0, 0]), v=np.zeros(3))
        assert s.validate() is False

    def test_inf_v_invalid(self) -> None:
        s = SimulationState(q=np.zeros(3), v=np.array([np.inf, 0, 0]))
        assert s.validate() is False

    def test_nan_a_invalid(self) -> None:
        s = SimulationState(q=np.zeros(3), v=np.zeros(3), a=np.array([np.nan, 0, 0]))
        assert s.validate() is False

    def test_nan_tau_invalid(self) -> None:
        s = SimulationState(q=np.zeros(3), v=np.zeros(3), tau=np.array([np.nan, 0, 0]))
        assert s.validate() is False

    def test_negative_time_invalid(self) -> None:
        s = SimulationState(q=np.zeros(3), v=np.zeros(3), time=-1.0)
        assert s.validate() is False

    def test_copy_is_deep(self) -> None:
        s = SimulationState(q=np.ones(2), v=np.ones(2), metadata={"engine": "test"})
        c = s.copy()
        c.q[0] = 99
        c.metadata["engine"] = "other"
        assert s.q[0] == 1.0
        assert s.metadata["engine"] == "test"


# ---------------------------------------------------------------------------
# StateManager undo/redo edges
# ---------------------------------------------------------------------------


class TestStateManagerHistory:
    def test_undo_redo_walks_history(self) -> None:
        m = StateManager(nq=2, nv=2, max_history=10)
        m.initialize(q0=np.array([1.0, 2.0]))
        m._state.q[:] = [3.0, 4.0]
        m._save_history()
        m._state.q[:] = [5.0, 6.0]
        m._save_history()

        assert m.can_undo()
        assert not m.can_redo()
        assert m.undo() is True
        np.testing.assert_array_equal(m.state.q, [3.0, 4.0])
        assert m.can_redo()
        assert m.redo() is True
        np.testing.assert_array_equal(m.state.q, [5.0, 6.0])

    def test_undo_at_start_returns_false(self) -> None:
        m = StateManager(nq=2, nv=2)
        m.initialize()
        assert m.undo() is False

    def test_redo_at_end_returns_false(self) -> None:
        m = StateManager(nq=2, nv=2)
        m.initialize()
        assert m.redo() is False

    def test_history_capped_at_max(self) -> None:
        m = StateManager(nq=1, nv=1, max_history=3)
        m.initialize()
        for _ in range(10):
            m._save_history()
        assert len(m._history) <= 3

    def test_reset_clears_history(self) -> None:
        m = StateManager(nq=1, nv=1)
        m.initialize(q0=np.array([5.0]))
        m._save_history()
        m.reset()
        assert m._history == []
        np.testing.assert_array_equal(m.state.q, np.zeros(1))

    def test_set_state_dimension_mismatch_raises(self) -> None:
        m = StateManager(nq=2, nv=2)
        with pytest.raises(ValueError, match="q dimension"):
            m.set_state(np.zeros(3), np.zeros(2))
        with pytest.raises(ValueError, match="v dimension"):
            m.set_state(np.zeros(2), np.zeros(5))

    def test_advance_time_updates_step_count(self) -> None:
        m = StateManager(nq=1, nv=1)
        m.initialize()
        m.advance_time(0.01)
        m.advance_time(0.01)
        assert m.state.step_count == 2
        assert m.state.time == pytest.approx(0.02)

    def test_get_state_returns_copies(self) -> None:
        m = StateManager(nq=2, nv=2)
        m.initialize(q0=np.array([1.0, 2.0]))
        q, v = m.get_state()
        q[0] = 99
        assert m.state.q[0] == 1.0


# ---------------------------------------------------------------------------
# EngineStateMixin
# ---------------------------------------------------------------------------


class _DummyEngine(EngineStateMixin):
    pass


class TestEngineStateMixin:
    def test_initial_state_is_uninitialized(self) -> None:
        e = _DummyEngine()
        assert e._get_lifecycle() == EngineLifecycleState.UNINITIALIZED

    def test_set_lifecycle_notifies_callbacks(self) -> None:
        e = _DummyEngine()
        observed: list[EngineLifecycleState] = []
        e.add_lifecycle_callback(observed.append)
        e._set_lifecycle(EngineLifecycleState.INITIALIZED)
        assert observed == [EngineLifecycleState.INITIALIZED]

    def test_callback_errors_are_swallowed(self) -> None:
        e = _DummyEngine()

        def boom(_: EngineLifecycleState) -> None:
            raise RuntimeError("kaboom")

        e.add_lifecycle_callback(boom)
        # Should not propagate
        e._set_lifecycle(EngineLifecycleState.INITIALIZED)

    def test_remove_callback(self) -> None:
        e = _DummyEngine()
        observed: list[EngineLifecycleState] = []
        e.add_lifecycle_callback(observed.append)
        e.remove_lifecycle_callback(observed.append)
        e._set_lifecycle(EngineLifecycleState.INITIALIZED)
        assert observed == []

    def test_remove_unknown_callback_is_noop(self) -> None:
        e = _DummyEngine()
        e.remove_lifecycle_callback(lambda _s: None)  # no error

    def test_require_lifecycle_raises_when_wrong_state(self) -> None:
        e = _DummyEngine()
        with pytest.raises(StateError):
            e._require_lifecycle(EngineLifecycleState.INITIALIZED, operation="step")

    def test_require_lifecycle_passes_when_state_matches(self) -> None:
        e = _DummyEngine()
        e._set_lifecycle(EngineLifecycleState.INITIALIZED)
        e._require_lifecycle(EngineLifecycleState.INITIALIZED, operation="step")


# ---------------------------------------------------------------------------
# ForceAccumulator
# ---------------------------------------------------------------------------


class TestForceAccumulator:
    def test_empty_returns_zero_vectors(self) -> None:
        acc = ForceAccumulator(nv=4)
        np.testing.assert_array_equal(acc.get_total_force(), np.zeros(3))
        np.testing.assert_array_equal(acc.get_total_torque(), np.zeros(3))
        np.testing.assert_array_equal(acc.get_total_generalized_force(), np.zeros(4))

    def test_add_force_default_zero_torque(self) -> None:
        acc = ForceAccumulator(nv=2)
        acc.add_force("g", np.array([0.0, 0.0, -9.81]))
        sources = acc.get_forces_by_source()
        assert "g" in sources
        np.testing.assert_array_equal(sources["g"].torque, np.zeros(3))

    def test_total_force_and_torque_sums(self) -> None:
        acc = ForceAccumulator(nv=2)
        acc.add_force("a", np.array([1.0, 0, 0]), torque=np.array([0, 1.0, 0]))
        acc.add_force("b", np.array([0.0, 2.0, 0]), torque=np.array([0, 0, 3.0]))
        np.testing.assert_allclose(acc.get_total_force(), [1.0, 2.0, 0.0])
        np.testing.assert_allclose(acc.get_total_torque(), [0.0, 1.0, 3.0])

    def test_add_generalized_force_dimension_check(self) -> None:
        acc = ForceAccumulator(nv=3)
        acc.add_generalized_force("m", np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(acc.get_total_generalized_force(), [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="tau dimension"):
            acc.add_generalized_force("bad", np.array([1.0, 2.0]))

    def test_get_forces_by_category_groups(self) -> None:
        acc = ForceAccumulator(nv=1)
        acc.add_force("g", np.zeros(3), category="gravity")
        acc.add_force("d", np.zeros(3), category="drag")
        acc.add_force("d2", np.zeros(3), category="drag")
        by_cat = acc.get_forces_by_category()
        assert set(by_cat) == {"gravity", "drag"}
        assert len(by_cat["drag"]) == 2

    def test_get_source_names_includes_generalized(self) -> None:
        acc = ForceAccumulator(nv=2)
        acc.add_force("g", np.zeros(3))
        acc.add_generalized_force("m", np.zeros(2))
        assert set(acc.get_source_names()) == {"g", "m"}

    def test_clear_removes_all(self) -> None:
        acc = ForceAccumulator(nv=2)
        acc.add_force("g", np.ones(3))
        acc.add_generalized_force("m", np.ones(2))
        acc.clear()
        np.testing.assert_array_equal(acc.get_total_force(), np.zeros(3))
        assert acc.get_source_names() == []

    def test_force_source_default_zero_arrays(self) -> None:
        src = ForceSource(name="x")
        np.testing.assert_array_equal(src.force, np.zeros(3))
        np.testing.assert_array_equal(src.torque, np.zeros(3))
        assert src.category == "external"
