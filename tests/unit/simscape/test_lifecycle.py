"""Unit tests for the SimscapeAdapter lifecycle helper (#4005)."""

from __future__ import annotations

import pytest
from src.engines.simscape._errors import SimscapeStateError
from src.engines.simscape._lifecycle import AdapterState, LifecycleGuard


@pytest.mark.unit
def test_initial_state_is_uninitialised() -> None:
    guard = LifecycleGuard()
    assert guard.state == AdapterState.UNINITIALIZED
    assert not guard.is_loaded()
    assert not guard.is_stopped()


@pytest.mark.unit
def test_lifecycle_uninitialised_to_loaded() -> None:
    guard = LifecycleGuard()
    guard.transition(AdapterState.LOADED, operation="load_from_path")
    assert guard.state == AdapterState.LOADED
    assert guard.is_loaded()


@pytest.mark.unit
def test_lifecycle_loaded_to_running_to_loaded_via_reset() -> None:
    guard = LifecycleGuard()
    guard.transition(AdapterState.LOADED, operation="load_from_path")
    guard.transition(AdapterState.RUNNING, operation="step")
    assert guard.is_loaded()  # RUNNING is also "loaded"
    guard.transition(AdapterState.LOADED, operation="reset")
    assert guard.state == AdapterState.LOADED


@pytest.mark.unit
def test_lifecycle_invalid_transition_raises() -> None:
    guard = LifecycleGuard()
    # Cannot go UNINITIALIZED -> RUNNING directly
    with pytest.raises(SimscapeStateError) as exc_info:
        guard.transition(AdapterState.RUNNING, operation="step")
    assert exc_info.value.operation == "step"
    assert exc_info.value.current_state == AdapterState.UNINITIALIZED.value


@pytest.mark.unit
def test_lifecycle_close_is_idempotent() -> None:
    guard = LifecycleGuard()
    guard.transition(AdapterState.STOPPED, operation="close")
    assert guard.is_stopped()
    # Re-running close is allowed (STOPPED -> STOPPED).
    guard.transition(AdapterState.STOPPED, operation="close")
    assert guard.is_stopped()


@pytest.mark.unit
def test_lifecycle_require_passes_when_state_matches() -> None:
    guard = LifecycleGuard()
    guard.transition(AdapterState.LOADED, operation="load_from_path")
    # Should not raise.
    guard.require(
        AdapterState.LOADED,
        AdapterState.RUNNING,
        operation="get_state",
    )


@pytest.mark.unit
def test_lifecycle_require_raises_when_state_mismatched() -> None:
    guard = LifecycleGuard()
    with pytest.raises(SimscapeStateError) as exc_info:
        guard.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="get_state",
        )
    assert exc_info.value.operation == "get_state"
    assert "loaded" in exc_info.value.required_state
