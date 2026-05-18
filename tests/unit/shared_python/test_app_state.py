"""Unit tests for src.shared.python.app_state package.

TDD: These tests were written BEFORE the implementation (RED → GREEN).
All tests focus on pure-data behaviour; Qt imports are avoided so the
suite runs headless in CI.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> Any:
    from src.shared.python.app_state import HistoryStore

    return HistoryStore()


def _make_engine() -> Any:
    from src.shared.python.app_state import DiagnosticEngine

    return DiagnosticEngine()


# ===========================================================================
# StateLogger singleton
# ===========================================================================


@pytest.mark.unit
def test_get_state_logger_is_singleton() -> None:
    """get_state_logger() must return the same object every call."""
    from src.shared.python.app_state import get_state_logger

    a = get_state_logger()
    b = get_state_logger()
    assert a is b


@pytest.mark.unit
def test_log_event_appends_app_event() -> None:
    """log_event appends an AppEvent to the internal HistoryStore."""
    from src.shared.python.app_state import get_state_logger

    logger = get_state_logger()
    before_len = len(logger.store)
    logger.log_event("test_action", {"key": "value"})
    assert len(logger.store) == before_len + 1
    last = logger.store.latest()
    assert last.type == "test_action"
    assert last.payload == {"key": "value"}


# ===========================================================================
# HistoryStore
# ===========================================================================


@pytest.mark.unit
def test_history_store_thread_safety() -> None:
    """5 threads × 200 events must all land without data corruption."""
    store = _make_store()
    errors: list[Exception] = []

    def _worker() -> None:
        try:
            for i in range(200):
                store.append_event("thread_event", {"i": i})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    # Store holds at most maxlen items (no crash)
    assert len(store) <= store.maxlen


@pytest.mark.unit
def test_history_store_ring_buffer() -> None:
    """Appending more than maxlen events keeps the store at maxlen."""
    from src.shared.python.app_state._history_store import HistoryStore

    store = HistoryStore(maxlen=5)
    for i in range(20):
        store.append_event("ev", {"i": i})
    assert len(store) == 5


@pytest.mark.unit
def test_history_store_as_json_valid() -> None:
    """as_json() must always return parseable JSON."""
    store = _make_store()
    store.append_event("startup", {"version": "1.0"})
    raw = store.as_json()
    parsed = json.loads(raw)
    assert isinstance(parsed, list)
    assert parsed[0]["type"] == "startup"


@pytest.mark.unit
def test_history_clear() -> None:
    """clear() must empty the store."""
    store = _make_store()
    store.append_event("ev", {})
    store.append_event("ev", {})
    store.clear()
    assert len(store) == 0


# ===========================================================================
# DiagnosticEngine
# ===========================================================================


@pytest.mark.unit
def test_diagnostic_engine_pass_check() -> None:
    """A check that returns True produces a PASS result."""
    from src.shared.python.app_state import DiagnosticEngine
    from src.shared.python.app_state._diagnostic import DiagnosticResult

    engine = DiagnosticEngine()
    engine.register_check("always_pass", lambda: True)
    results = engine.run_checks()
    by_name = {r.name: r for r in results}
    assert by_name["always_pass"].status == "PASS"


@pytest.mark.unit
def test_diagnostic_engine_fail_check() -> None:
    """A check that returns False produces a FAIL result."""
    from src.shared.python.app_state import DiagnosticEngine

    engine = DiagnosticEngine()
    engine.register_check("always_fail", lambda: False)
    results = engine.run_checks()
    by_name = {r.name: r for r in results}
    assert by_name["always_fail"].status == "FAIL"


@pytest.mark.unit
def test_diagnostic_engine_exception_gives_fail_not_raise() -> None:
    """A check that raises must not propagate — it becomes FAIL."""
    from src.shared.python.app_state import DiagnosticEngine

    engine = DiagnosticEngine()

    def _boom() -> bool:
        raise RuntimeError("simulated failure")

    engine.register_check("exploding_check", _boom)
    results = engine.run_checks()  # must not raise
    by_name = {r.name: r for r in results}
    assert by_name["exploding_check"].status == "FAIL"
    assert "simulated failure" in by_name["exploding_check"].message


# ===========================================================================
# agent_context
# ===========================================================================


@pytest.mark.unit
def test_agent_context_keys() -> None:
    """agent_context() dict must contain 'events', 'last_diagnostics', 'summary'."""
    from src.shared.python.app_state import agent_context

    store = _make_store()
    store.append_event("boot", {"ok": True})
    ctx = agent_context(store)
    assert "events" in ctx
    assert "last_diagnostics" in ctx
    assert "summary" in ctx


@pytest.mark.unit
def test_agent_context_max_events() -> None:
    """agent_context() must honour max_events parameter."""
    from src.shared.python.app_state import agent_context

    store = _make_store()
    for i in range(100):
        store.append_event("ev", {"i": i})
    ctx = agent_context(store, max_events=10)
    assert len(ctx["events"]) == 10
