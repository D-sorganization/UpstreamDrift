"""Tests for src.engines.simscape._lifecycle."""

from __future__ import annotations

import pytest

from src.engines.simscape._errors import SimscapeStateError
from src.engines.simscape._lifecycle import AdapterState, LifecycleGuard


def test_initial_state_uninitialized() -> None:
    g = LifecycleGuard()
    assert g.state is AdapterState.UNINITIALIZED
    assert not g.is_loaded()
    assert not g.is_stopped()


def test_load_then_run_then_close_path() -> None:
    g = LifecycleGuard()
    g.transition(AdapterState.LOADED, operation="load")
    assert g.is_loaded()
    g.transition(AdapterState.RUNNING, operation="step")
    assert g.is_loaded()
    g.transition(AdapterState.RUNNING, operation="step")  # running->running OK
    g.transition(AdapterState.LOADED, operation="reset")
    g.transition(AdapterState.STOPPED, operation="close")
    assert g.is_stopped()


def test_close_idempotent_from_any_state() -> None:
    g = LifecycleGuard()
    # uninitialized -> stopped
    g.transition(AdapterState.STOPPED, operation="close")
    # stopped -> stopped
    g.transition(AdapterState.STOPPED, operation="close")
    assert g.is_stopped()


def test_illegal_transition_raises_state_error() -> None:
    g = LifecycleGuard()
    with pytest.raises(SimscapeStateError) as exc:
        g.transition(AdapterState.RUNNING, operation="step")
    assert exc.value.operation == "step"
    assert exc.value.current_state == "uninitialized"


def test_require_passes_for_allowed_state() -> None:
    g = LifecycleGuard()
    g.transition(AdapterState.LOADED, operation="load")
    g.require(AdapterState.LOADED, AdapterState.RUNNING, operation="get_state")


def test_require_raises_for_disallowed_state() -> None:
    g = LifecycleGuard()
    with pytest.raises(SimscapeStateError) as exc:
        g.require(AdapterState.LOADED, operation="get_state")
    assert "uninitialized" in str(exc.value)
    assert "loaded" in str(exc.value)


def test_state_enum_values() -> None:
    assert AdapterState.UNINITIALIZED.value == "uninitialized"
    assert AdapterState.LOADED.value == "loaded"
    assert AdapterState.RUNNING.value == "running"
    assert AdapterState.STOPPED.value == "stopped"
