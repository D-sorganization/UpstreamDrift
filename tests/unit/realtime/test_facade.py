"""Unit tests for the realtime facade in ``src.shared.python.realtime``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.shared.python.realtime as facade_module
from src.shared.python.realtime import (
    CHANNEL_REGISTRY,
    Subscription,
    publish,
    register_channel,
    subscribe,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_transport() -> None:
    """Reset the module-level transport singleton between tests."""
    facade_module._TRANSPORT = None
    yield
    facade_module._TRANSPORT = None


@pytest.fixture()
def patched_transport() -> MagicMock:
    """Return a mock FileTransport wired into the facade."""
    transport = MagicMock()
    transport.publish = MagicMock()
    token_counter = 0

    def _subscribe(_channel, _callback):
        nonlocal token_counter
        token_counter += 1
        return token_counter

    transport.subscribe = MagicMock(side_effect=_subscribe)
    transport.unsubscribe = MagicMock()
    facade_module._TRANSPORT = transport
    return transport


class TestChannelRegistry:
    def test_register_new_channel(self) -> None:
        register_channel("test/chan", "Test channel")
        assert "test/chan" in CHANNEL_REGISTRY
        assert CHANNEL_REGISTRY["test/chan"].name == "test/chan"
        assert CHANNEL_REGISTRY["test/chan"].description == "Test channel"
        assert CHANNEL_REGISTRY["test/chan"].owner_tool_id is None

    def test_register_existing_channel_same_description_is_noop(self) -> None:
        register_channel("test/chan", "Test channel")
        register_channel("test/chan", "Test channel")  # should not raise
        assert len([k for k in CHANNEL_REGISTRY if k == "test/chan"]) == 1

    def test_register_existing_channel_different_description_raises(self) -> None:
        register_channel("test/chan", "Test channel")
        with pytest.raises(ValueError):
            register_channel("test/chan", "Different desc")

    def test_register_empty_channel_raises(self) -> None:
        with pytest.raises(ValueError):
            register_channel("", "desc")

    def test_builtin_pose_canonical_channel_is_registered(self) -> None:
        assert "pose/canonical" in CHANNEL_REGISTRY
        info = CHANNEL_REGISTRY["pose/canonical"]
        assert info.owner_tool_id == "pose_studio"
        assert "Pose Studio" in info.description


class TestPublish:
    def test_publish_calls_transport(self, patched_transport: MagicMock) -> None:
        publish("test/chan", {"v": 1})
        patched_transport.publish.assert_called_once_with("test/chan", {"v": 1})

    def test_publish_with_explicit_file_transport(
        self, patched_transport: MagicMock
    ) -> None:
        publish("test/chan", {"v": 1}, transport="file")
        patched_transport.publish.assert_called_once_with("test/chan", {"v": 1})

    def test_publish_with_ws_transport_falls_back_to_file(
        self, patched_transport: MagicMock
    ) -> None:
        """When transport='ws' is requested but not wired, fall back to file."""
        publish("test/chan", {"v": 1}, transport="ws")
        patched_transport.publish.assert_called_once_with("test/chan", {"v": 1})

    def test_publish_with_invalid_channel_logs_warning(
        self, caplog: pytest.LogCaptureFixture, patched_transport: MagicMock
    ) -> None:
        publish("", {"v": 1})
        patched_transport.publish.assert_not_called()
        assert "invalid channel" in caplog.text.lower()

    def test_publish_preserves_transport_parameter_signature(self) -> None:
        """Regression test for #5058: transport= is preserved in the signature."""
        import inspect

        sig = inspect.signature(publish)
        param_names = list(sig.parameters.keys())
        assert "transport" in param_names, (
            "publish() must preserve the 'transport' parameter "
            "(#5058 — Preserve realtime facade transport parameter)"
        )


class TestSubscribe:
    def test_subscribe_returns_subscription(self, patched_transport: MagicMock) -> None:
        cb = MagicMock()
        sub = subscribe("test/chan", cb)
        assert isinstance(sub, Subscription)
        assert sub.channel == "test/chan"
        assert not sub._closed

    def test_subscribe_invalid_channel_raises(self) -> None:
        with pytest.raises(ValueError):
            subscribe("", MagicMock())

    def test_subscribe_non_callable_raises(self) -> None:
        with pytest.raises(TypeError):
            subscribe("test/chan", "not_callable")  # type: ignore[arg-type]

    def test_unsubscribe_is_idempotent(self, patched_transport: MagicMock) -> None:
        sub = subscribe("test/chan", MagicMock())
        sub.unsubscribe()
        assert sub._closed is True
        # Second call should not raise
        sub.unsubscribe()
        assert sub._closed is True

    def test_unsubscribe_calls_transport(self, patched_transport: MagicMock) -> None:
        cb = MagicMock()
        sub = subscribe("test/chan", cb)
        sub.unsubscribe()
        patched_transport.unsubscribe.assert_called_once()


class TestRealFileTransportRoundTrip:
    def test_round_trip_via_tmp_path(self, tmp_path: Path) -> None:
        """End-to-end: publish writes a file; the file contains the payload."""
        os.environ["UPSTREAM_DRIFT_REALTIME_ROOT"] = str(tmp_path)
        facade_module._TRANSPORT = None
        try:
            publish("target/active", {"answer": 42}, transport="file")
            expected = tmp_path / "target__active.json"
            assert expected.exists()
            assert json.loads(expected.read_text(encoding="utf-8")) == {"answer": 42}
        finally:
            os.environ.pop("UPSTREAM_DRIFT_REALTIME_ROOT", None)
            facade_module._TRANSPORT = None
