"""REST provenance tests for POST /analysis/cross-engine (#8817).

The API must never silently substitute a 2-DOF stub for a known engine
name: the completed task payload declares each engine's actual backend,
and callers can refuse stub substitution outright.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.api.routes.cross_engine import (
    CrossEnginePerturbationConfig,
    CrossEngineStudyRequest,
    _run_study_background,
)
from src.shared.python.analysis import cross_engine as ce

pytestmark = pytest.mark.unit


class _FakeTaskManager:
    """Minimal task-manager double recording lifecycle calls."""

    def __init__(self) -> None:
        self.completed: dict[str, Any] = {}
        self.failed: dict[str, str] = {}

    def update_progress(self, task_id: str, progress: int) -> None:
        pass

    def mark_completed(self, task_id: str, result: dict[str, Any]) -> None:
        self.completed[task_id] = result

    def mark_failed(self, task_id: str, message: str) -> None:
        self.failed[task_id] = message


_FAST_CONFIG = CrossEnginePerturbationConfig(t_end=0.1, dt=0.01, n_trials=2, seed=1)


def test_result_payload_declares_stub_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known engine name backed by a stub must be declared in the result."""
    monkeypatch.setattr(ce, "try_build_real_engine", lambda _name: None)
    manager = _FakeTaskManager()
    request = CrossEngineStudyRequest(engines=["drake"], config=_FAST_CONFIG)
    _run_study_background("task-1", request, manager)
    assert "task-1" in manager.completed, manager.failed
    result = manager.completed["task-1"]
    assert result["engines"]["drake"]["backend"] == ce.BACKEND_STUB
    assert result["stubbed_engines"] == ["drake"]


def test_request_can_refuse_stub_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """allow_stub_substitution=False must fail the task, naming the engine."""
    monkeypatch.setattr(ce, "try_build_real_engine", lambda _name: None)
    manager = _FakeTaskManager()
    request = CrossEngineStudyRequest(
        engines=["mujoco"],
        config=_FAST_CONFIG,
        allow_stub_substitution=False,
    )
    _run_study_background("task-2", request, manager)
    assert "task-2" not in manager.completed
    assert "mujoco" in manager.failed["task-2"]


def test_real_engine_backend_is_declared_real(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the real engine builds, the payload must say backend=real."""
    monkeypatch.setattr(
        ce, "try_build_real_engine", lambda _name: ce.StubEngine("impostor")
    )
    manager = _FakeTaskManager()
    request = CrossEngineStudyRequest(engines=["mujoco"], config=_FAST_CONFIG)
    _run_study_background("task-3", request, manager)
    result = manager.completed["task-3"]
    assert result["engines"]["mujoco"]["backend"] == ce.BACKEND_REAL
    assert result["stubbed_engines"] == []


def test_unknown_engine_still_rejected() -> None:
    """Existing typo-rejection contract must be preserved."""
    with pytest.raises(ValueError, match="Unknown engine"):
        CrossEngineStudyRequest(engines=["not_an_engine"])
