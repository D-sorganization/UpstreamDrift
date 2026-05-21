"""Tests for the realtime public API facade (api.py)."""

from __future__ import annotations

import threading
from pathlib import Path

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
    yield
    transport = api_mod._TRANSPORT
    if transport is not None and hasattr(transport, "shutdown"):
        transport.shutdown()
    monkeypatch.setattr(api_mod, "_TRANSPORT", None)


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

    def test_publish_non_file_transport_falls_back_to_file(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        # ws is not wired; should fall back silently to file
        publish("scope/topic", {"v": 1}, transport="ws")
        # The file under our isolated root should now exist
        files = list(tmp_path.glob("scope__topic.jsonl"))
        assert len(files) == 1

    def test_publish_uses_env_transport(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("REALTIME_TRANSPORT", "file")
        publish("scope/topic", {"v": 1})
        assert (tmp_path / "scope__topic.jsonl").exists()

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


class TestSubscriptionDataclass:
    def test_unsubscribe_idempotent(self) -> None:
        calls = []

        class Trans:
            def unsubscribe(self, token):
                calls.append(token)

        sub = Subscription(channel="a/b", _transport=Trans(), _token=5)
        sub.unsubscribe()
        sub.unsubscribe()
        assert calls == [5]

    def test_unsubscribe_no_transport_is_safe(self) -> None:
        sub = Subscription(channel="a/b", _transport=None, _token=-1)
        sub.unsubscribe()  # no raise

    def test_unsubscribe_swallows_transport_exception(self) -> None:
        class BoomTrans:
            def unsubscribe(self, token):
                raise RuntimeError("nope")

        sub = Subscription(channel="a/b", _transport=BoomTrans(), _token=1)
        sub.unsubscribe()  # must not raise


def test_get_transport_is_cached(tmp_path: Path) -> None:
    a = api_mod._get_transport()
    b = api_mod._get_transport()
    assert a is b
