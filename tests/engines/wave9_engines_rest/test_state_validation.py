"""Validation-error coverage for state.py after duplicate guard cleanup."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.engines.common.state import (
    EngineLifecycleState,
    EngineStateMixin,
    ForceAccumulator,
    StateManager,
)
from src.shared.python.core.contracts import PreconditionError


pytestmark = pytest.mark.unit


def _run_optimized_state_contract(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )


class TestStateManagerNoneGuards:
    def test_nq_none_rejected(self) -> None:
        with pytest.raises(PreconditionError):
            StateManager(nq=None, nv=3)  # type: ignore[arg-type]

    def test_nv_none_rejected(self) -> None:
        # nq positive but nv None: contract guard fires after first check
        with pytest.raises(PreconditionError):
            StateManager(nq=3, nv=None)  # type: ignore[arg-type]

    def test_advance_time_dt_none_rejected(self) -> None:
        m = StateManager(nq=2, nv=2)
        with pytest.raises(PreconditionError):
            m.advance_time(None)  # type: ignore[arg-type]


class TestStateManagerOptimizedRuntimeGuards:
    def test_constructor_rejects_non_positive_dimensions_under_optimized_python(
        self,
    ) -> None:
        result = _run_optimized_state_contract(
            """
from src.shared.python.core.contracts import ContractLevel, set_contract_level
from src.engines.common.state import StateManager
set_contract_level(ContractLevel.OFF)
for nq, nv in ((0, 6), (-1, 6), (6, 0), (6, -1)):
    try:
        StateManager(nq=nq, nv=nv)
    except ValueError:
        continue
    raise AssertionError(f"accepted nq={nq}, nv={nv}")
"""
        )
        assert result.returncode == 0, result.stderr

    def test_advance_time_rejects_non_positive_dt_under_optimized_python(
        self,
    ) -> None:
        result = _run_optimized_state_contract(
            """
from src.shared.python.core.contracts import ContractLevel, set_contract_level
from src.engines.common.state import StateManager
set_contract_level(ContractLevel.OFF)
manager = StateManager(nq=2, nv=2)
for dt in (0.0, -0.01):
    try:
        manager.advance_time(dt)
    except ValueError:
        continue
    raise AssertionError(f"accepted dt={dt}")
"""
        )
        assert result.returncode == 0, result.stderr

    def test_add_force_rejects_invalid_names_under_optimized_python(self) -> None:
        result = _run_optimized_state_contract(
            """
import numpy as np
from src.shared.python.core.contracts import ContractLevel, set_contract_level
from src.engines.common.state import ForceAccumulator


set_contract_level(ContractLevel.OFF)
accumulator = ForceAccumulator(nv=3)
for name in ("", 123):
    try:
        accumulator.add_force(name, np.zeros(3))
    except ValueError:
        continue
    raise AssertionError(f"accepted name={name!r}")
"""
        )
        assert result.returncode == 0, result.stderr


class TestEngineStateMixinSetLifecycleNoneGuard:
    def test_set_lifecycle_none_rejected(self) -> None:
        class E(EngineStateMixin):
            pass

        e = E()
        e.__init__()  # type: ignore[misc]
        with pytest.raises(ValueError, match="state must be provided"):
            e._set_lifecycle(None)  # type: ignore[arg-type]

    def test_set_lifecycle_valid_transitions(self) -> None:
        class E(EngineStateMixin):
            pass

        e = E()
        e.__init__()  # type: ignore[misc]
        e._set_lifecycle(EngineLifecycleState.INITIALIZED)
        assert e._get_lifecycle() == EngineLifecycleState.INITIALIZED


class TestForceAccumulatorNvGuard:
    def test_nv_none_rejected(self) -> None:
        with pytest.raises(PreconditionError):
            ForceAccumulator(nv=None)  # type: ignore[arg-type]

    def test_add_generalized_force_dimension_mismatch(self) -> None:
        acc = ForceAccumulator(nv=4)
        with pytest.raises(ValueError, match="dimension mismatch"):
            acc.add_generalized_force("muscle", np.zeros(3))

    def test_add_generalized_force_accepts_correct_dim(self) -> None:
        acc = ForceAccumulator(nv=3)
        acc.add_generalized_force("muscle", np.array([1.0, 2.0, 3.0]))
        total = acc.get_total_generalized_force()
        np.testing.assert_array_equal(total, [1.0, 2.0, 3.0])


class TestStateManagerHistoryEdges:
    def test_history_eviction_when_full(self) -> None:
        m = StateManager(nq=1, nv=1, max_history=2)
        m.initialize(np.array([0.0]))
        m._save_history()  # second save
        m._save_history()  # third — evicts oldest
        assert len(m._history) == 2

    def test_undo_redo_round_trip(self) -> None:
        m = StateManager(nq=1, nv=1, max_history=10)
        m.initialize(np.array([0.0]))
        m.set_state(np.array([1.0]), np.array([0.0]))
        m._save_history()
        m.set_state(np.array([2.0]), np.array([0.0]))
        m._save_history()
        assert m.can_undo()
        assert m.undo() is True
        np.testing.assert_array_equal(m._state.q, [1.0])
        assert m.can_redo()
        assert m.redo() is True

    def test_undo_at_start_returns_false(self) -> None:
        m = StateManager(nq=1, nv=1)
        m.initialize()
        assert m.undo() is False

    def test_redo_at_end_returns_false(self) -> None:
        m = StateManager(nq=1, nv=1)
        m.initialize()
        assert m.redo() is False
