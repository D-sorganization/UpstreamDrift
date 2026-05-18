"""Tests for WorkspaceRegistry.subscribe / unsubscribe (Issue #5616).

TDD: these tests are written first, before the implementation.
"""

from __future__ import annotations

import pytest

from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry

pytestmark = pytest.mark.unit


def test_registry_subscribe_fires_on_set_variable() -> None:
    """Precondition: subscribe returns Subscription; fires on set_variable."""
    registry = WorkspaceRegistry()
    fired: list[tuple[str, object]] = []
    registry.subscribe(lambda name, value: fired.append((name, value)))
    registry.set_variable("x", 42)
    assert fired == [("x", 42)]


def test_registry_subscribe_fires_on_delete_variable() -> None:
    """Subscriber fires on delete_variable with value=None."""
    registry = WorkspaceRegistry()
    registry.set_variable("x", 10)
    events: list[tuple[str, object]] = []
    registry.subscribe(lambda name, value: events.append((name, value)))
    registry.delete_variable("x")
    assert ("x", None) in events


def test_registry_unsubscribe_stops_firing() -> None:
    """Postcondition: Subscription.dispose() removes the callback."""
    registry = WorkspaceRegistry()
    fired: list[str] = []
    sub = registry.subscribe(lambda n, v: fired.append(n))
    sub.dispose()
    registry.set_variable("x", 1)
    assert fired == []


def test_registry_subscribe_multiple_callbacks() -> None:
    """Both subscribers fire independently."""
    registry = WorkspaceRegistry()
    a: list[str] = []
    b: list[str] = []
    registry.subscribe(lambda n, v: a.append(n))
    registry.subscribe(lambda n, v: b.append(n))
    registry.set_variable("y", 2)
    assert a == ["y"]
    assert b == ["y"]


def test_subscribe_none_raises_type_error() -> None:
    """DbC: subscribe(None) raises TypeError."""
    with pytest.raises(TypeError, match="callable"):
        WorkspaceRegistry().subscribe(None)  # type: ignore[arg-type]


def test_subscribe_non_callable_raises_type_error() -> None:
    """DbC: subscribe with a non-callable raises TypeError."""
    with pytest.raises(TypeError, match="callable"):
        WorkspaceRegistry().subscribe("not_callable")  # type: ignore[arg-type]


def test_set_variable_forwards_to_registry_set() -> None:
    """set_variable stores the value in the registry."""
    registry = WorkspaceRegistry()
    registry.set_variable("z", 99)
    assert registry.get_variable("z") == 99


def test_get_variable_returns_default_when_absent() -> None:
    """get_variable returns None when name not set."""
    registry = WorkspaceRegistry()
    assert registry.get_variable("missing") is None


def test_no_reentrancy_loop_on_subscribe_firing() -> None:
    """Invariant: set_variable from inside a callback does not loop infinitely."""
    registry = WorkspaceRegistry()
    call_count = 0

    def callback(name: str, value: object) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Second set_variable from within callback
            registry.set_variable("inner", 0)

    registry.subscribe(callback)
    registry.set_variable("outer", 1)
    # callback fires for "outer" (count=1) then for "inner" (count=2)
    # it must not recurse indefinitely
    assert call_count == 2  # noqa: PLR2004
