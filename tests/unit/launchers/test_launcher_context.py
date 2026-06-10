"""Headless tests for the launcher embedded-tool context (#7210)."""

from __future__ import annotations

import pytest

from src.shared.python.launcher_embed.context import InMemoryLauncherContext

pytestmark = pytest.mark.unit


def test_emit_dispatches_to_subscribers_with_snapshot_order() -> None:
    context = InMemoryLauncherContext()
    events: list[tuple[str, dict[str, object]]] = []

    context.subscribe("engine.switched", lambda event: events.append(("a", event)))
    context.subscribe("engine.switched", lambda event: events.append(("b", event)))

    context.emit("engine.switched", {"engine": "mujoco"})

    assert events == [
        ("a", {"engine": "mujoco"}),
        ("b", {"engine": "mujoco"}),
    ]


def test_unsubscribe_is_idempotent_and_stops_future_dispatch() -> None:
    context = InMemoryLauncherContext()
    seen: list[dict[str, object]] = []
    unsubscribe = context.subscribe("model.loaded", seen.append)

    unsubscribe()
    unsubscribe()
    context.emit("model.loaded", {"path": "demo.urdf"})

    assert seen == []


def test_emit_uses_snapshot_when_callback_unsubscribes_peer() -> None:
    context = InMemoryLauncherContext()
    seen: list[str] = []

    def first(_: dict[str, object]) -> None:
        seen.append("first")
        unsubscribe_second()

    def second(_: dict[str, object]) -> None:
        seen.append("second")

    context.subscribe("tab.opened", first)
    unsubscribe_second = context.subscribe("tab.opened", second)

    context.emit("tab.opened", {"tool_id": "a"})
    context.emit("tab.opened", {"tool_id": "b"})

    assert seen == ["first", "second", "first"]


def test_reentrant_emit_dispatches_nested_event_without_looping() -> None:
    context = InMemoryLauncherContext()
    seen: list[str] = []

    def callback(event: dict[str, object]) -> None:
        tool_id = str(event["tool_id"])
        seen.append(tool_id)
        if tool_id == "outer":
            context.emit("tab.opened", {"tool_id": "inner"})

    context.subscribe("tab.opened", callback)

    context.emit("tab.opened", {"tool_id": "outer"})

    assert seen == ["outer", "inner"]


def test_set_value_stores_value_and_emits_keyed_change_event() -> None:
    context = InMemoryLauncherContext()
    seen: list[dict[str, object]] = []
    context.subscribe("value_changed:active_model", seen.append)

    previous = context.set_value("active_model", "demo.urdf")

    assert previous is None
    assert context.get_value("active_model") == "demo.urdf"
    assert context.list() == ["active_model"]
    assert seen == [
        {
            "key": "active_model",
            "value": "demo.urdf",
            "previous": None,
            "existed": False,
        }
    ]


def test_set_value_returns_previous_value() -> None:
    context = InMemoryLauncherContext()
    context.set_value("active_engine", "mujoco")

    previous = context.set_value("active_engine", "pinocchio")

    assert previous == "mujoco"
    assert context.get_value("active_engine") == "pinocchio"


@pytest.mark.parametrize("event_type", ["", "   ", None])
def test_emit_rejects_invalid_event_type(event_type: str | None) -> None:
    context = InMemoryLauncherContext()

    with pytest.raises(ValueError, match="event_type must be a non-empty string"):
        context.emit(event_type, {})  # type: ignore[arg-type]


@pytest.mark.parametrize("key", ["", "   ", None])
def test_set_value_rejects_invalid_key(key: str | None) -> None:
    context = InMemoryLauncherContext()

    with pytest.raises(ValueError, match="key must be a non-empty string"):
        context.set_value(key, 1)  # type: ignore[arg-type]


def test_subscribe_rejects_non_callable() -> None:
    context = InMemoryLauncherContext()

    with pytest.raises(TypeError, match="callback must be callable"):
        context.subscribe("tab.closed", "not-callable")  # type: ignore[arg-type]
