"""Tests for the realtime public API facade (api.py)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.shared.python.realtime import api as api_mod
from src.shared.python.realtime.api import (
    CHANNEL_REGISTRY,
    Subscription,
    publish,
    register_channel,
    subscribe,
)


@pytest.fixture(autouse=True)
def _isolate_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a fresh module-global transport rooted at tmp_path per test."""
    monkeypatch.setenv("REALTIME_FILE_ROOT", str(tmp_path))
    monkeypatch.setattr(api_mod, "_TRANSPORT", None)
    monkeypatch.setattr(api_mod, "_WS_TRANSPORT", None)
    yield
    transport = api_mod._TRANSPORT
    if transport is not None and hasattr(transport, "shutdown"):
        transport.shutdown()
    monkeypatch.setattr(api_mod, "_TRANSPORT", None)
    monkeypatch.setattr(api_mod, "_WS_TRANSPORT", None)


@pytest.fixture()
def patched_ws_transport(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Wire a mock in as the "ws" transport.

    Never let a test construct the real :class:`WSPubSub`: doing so
    autostarts a background server (Rust in-process, or a spawned
    uvicorn process), which is out of place in a unit test.
    """
    transport = MagicMock()
    transport.publish = MagicMock()
    sub_handle = MagicMock()
    sub_handle.unsubscribe = MagicMock()
    transport.subscribe = MagicMock(return_value=sub_handle)
    monkeypatch.setattr(api_mod, "_WS_TRANSPORT", transport)
    return transport


class TestRegisterChannel:
    def test_register_new_channel(self) -> None:
        register_channel("tests_api/new_one", "desc", owner_tool_id="t")
        assert CHANNEL_REGISTRY["tests_api/new_one"].description == "desc"
        assert CHANNEL_REGISTRY["tests_api/new_one"].owner_tool_id == "t"

    def test_register_same_descriptor_is_noop(self) -> None:
        register_channel("tests_api/idem", "d")
        register_channel("tests_api/idem", "d")  # no raise

    def test_re_register_with_different_descriptor_raises(self) -> None:
        register_channel("tests_api/conflict", "d1")
        with pytest.raises(ValueError, match="different descriptor"):
            register_channel("tests_api/conflict", "d2")

    def test_re_register_with_different_owner_raises(self) -> None:
        register_channel("tests_api/owner", "d", owner_tool_id="a")
        with pytest.raises(ValueError, match="different descriptor"):
            register_channel("tests_api/owner", "d", owner_tool_id="b")

    @pytest.mark.parametrize("bad", ["", "   ", None, 1, []])
    def test_invalid_name_raises(self, bad) -> None:
        with pytest.raises(ValueError):
            register_channel(bad, "x")  # type: ignore[arg-type]

    def test_pose_canonical_preregistered(self) -> None:
        assert "pose/canonical" in CHANNEL_REGISTRY


class TestPublish:
    def test_publish_invalid_channel_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        publish("", {"x": 1})  # logged + returned, no raise
        publish(None, {"x": 1})  # type: ignore[arg-type]
        publish("   ", {"x": 1})

    def test_publish_ws_transport_routes_to_ws(
        self, patched_ws_transport: MagicMock, tmp_path: Path
    ) -> None:
        """transport='ws' is wired (issue #8869): must reach the ws
        transport and must NOT write to the file transport's log."""
        publish("scope/topic", {"v": 1}, transport="ws")
        patched_ws_transport.publish.assert_called_once_with("scope/topic", {"v": 1})
        assert not list(tmp_path.glob("scope__topic.jsonl"))

    def test_publish_env_ws_transport_routes_to_ws(
        self, patched_ws_transport: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REALTIME_TRANSPORT", "ws")
        publish("scope/topic", {"v": 1})
        patched_ws_transport.publish.assert_called_once_with("scope/topic", {"v": 1})

    def test_publish_uses_env_transport(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REALTIME_TRANSPORT", "file")
        publish("scope/topic", {"v": 1})
        assert (tmp_path / "scope__topic.jsonl").exists()

    def test_publish_unsupported_transport_raises(self) -> None:
        """A config error must fail loudly, never fall back (#8869)."""
        with pytest.raises(ValueError, match="unsupported realtime transport"):
            publish("scope/topic", {"v": 1}, transport="carrier-pigeon")

    def test_publish_unsupported_env_transport_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REALTIME_TRANSPORT", "carrier-pigeon")
        with pytest.raises(ValueError, match="unsupported realtime transport"):
            publish("scope/topic", {"v": 1})

    def test_publish_swallows_transport_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Boom:
            def publish(self, channel, payload):
                raise RuntimeError("nope")

        monkeypatch.setattr(api_mod, "_TRANSPORT", Boom())
        publish("scope/topic", {"v": 1})  # must not raise


class TestSubscribe:
    def test_subscribe_invalid_channel_raises(self) -> None:
        with pytest.raises(ValueError):
            subscribe("", lambda p: None)
        with pytest.raises(ValueError):
            subscribe("   ", lambda p: None)

    def test_subscribe_non_callable_raises(self) -> None:
        with pytest.raises(TypeError):
            subscribe("scope/topic", 123)  # type: ignore[arg-type]

    def test_subscribe_and_publish_roundtrip(self, tmp_path: Path) -> None:
        received: list = []
        evt = threading.Event()

        def cb(payload) -> None:
            received.append(payload)
            evt.set()

        sub = subscribe("scope/topic", cb)
        try:
            assert isinstance(sub, Subscription)
            publish("scope/topic", {"v": 42})
            assert evt.wait(2.0)
            assert received[-1] == {"v": 42}
        finally:
            sub.unsubscribe()

    def test_subscribe_transport_failure_returns_inert_sub(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom() -> object:
            raise RuntimeError("no transport")

        monkeypatch.setattr(api_mod, "_get_transport", boom)
        sub = subscribe("scope/topic", lambda p: None)
        assert sub._closed is True
        # unsubscribe must be idempotent and not raise
        sub.unsubscribe()
        sub.unsubscribe()

    def test_subscribe_ws_transport_routes_to_ws(
        self, patched_ws_transport: MagicMock
    ) -> None:
        cb = lambda p: None  # noqa: E731
        sub = subscribe("scope/topic", cb, transport="ws")
        patched_ws_transport.subscribe.assert_called_once_with("scope/topic", cb)
        sub.unsubscribe()
        patched_ws_transport.subscribe.return_value.unsubscribe.assert_called_once()

    def test_subscribe_unsupported_transport_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported realtime transport"):
            subscribe("scope/topic", lambda p: None, transport="carrier-pigeon")


class TestSubscriptionDataclass:
    def test_unsubscribe_idempotent(self) -> None:
        calls = []

        sub = Subscription(channel="a/b", _unsubscribe_fn=lambda: calls.append(5))
        sub.unsubscribe()
        sub.unsubscribe()
        assert calls == [5]

    def test_unsubscribe_no_transport_is_safe(self) -> None:
        sub = Subscription(channel="a/b", _unsubscribe_fn=None)
        sub.unsubscribe()  # no raise

    def test_unsubscribe_swallows_transport_exception(self) -> None:
        def boom() -> None:
            raise RuntimeError("nope")

        sub = Subscription(channel="a/b", _unsubscribe_fn=boom)
        sub.unsubscribe()  # must not raise


def test_get_transport_is_cached(tmp_path: Path) -> None:
    a = api_mod._get_transport()
    b = api_mod._get_transport()
    assert a is b
