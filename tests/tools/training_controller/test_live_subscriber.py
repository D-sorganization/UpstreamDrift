"""Tests for :mod:`training_controller.live_subscriber`.

The realtime facade is patched with an in-memory stub so we exercise
payload decoding and callback dispatch without touching the file
transport.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from training import TrainingStatus
from training.metrics import MetricKind, TrainingMetric
from training.persistence import training_metric_to_dict
from training.runtime.progress_sinks import training_channel_for
from src.tools.training_controller.live_subscriber import (
    TrainingJobLiveSubscriber,
)


pytestmark = pytest.mark.unit


# --------------------------------------------------------------- stub realtime


class _StubSubscription:
    """Stand-in for :class:`src.shared.python.realtime.api.Subscription`."""

    def __init__(self, on_unsubscribe: Callable[[], None]) -> None:
        self._on_unsubscribe = on_unsubscribe
        self.unsubscribed = False

    def unsubscribe(self) -> None:
        self.unsubscribed = True
        self._on_unsubscribe()


class _StubRealtime:
    """Minimal stand-in for :mod:`src.shared.python.realtime`."""

    def __init__(self) -> None:
        self.registered: list[tuple[str, str, str | None]] = []
        self.subscriptions: dict[str, list[Callable[[Any], None]]] = {}
        self._lock = threading.Lock()
        self._next_token = 0
        self._token_to_channel: dict[int, str] = {}

    def register_channel(
        self,
        name: str,
        description: str = "",
        owner_tool_id: str | None = None,
    ) -> None:
        self.registered.append((name, description, owner_tool_id))

    def subscribe(
        self,
        channel: str,
        callback: Callable[[Any], None],
    ) -> _StubSubscription:
        with self._lock:
            self.subscriptions.setdefault(channel, []).append(callback)
            token = self._next_token
            self._token_to_channel[token] = channel
            self._next_token += 1

        def _unsubscribe() -> None:
            with self._lock:
                callbacks = self.subscriptions.get(channel, [])
                try:
                    callbacks.remove(callback)
                except ValueError:
                    return

        return _StubSubscription(on_unsubscribe=_unsubscribe)

    def publish(self, channel: str, payload: Any) -> None:
        with self._lock:
            callbacks = tuple(self.subscriptions.get(channel, ()))
        for cb in callbacks:
            cb(payload)


@pytest.fixture
def stub_realtime() -> Any:
    """Install a stub :mod:`src.shared.python.realtime` module."""

    stub = _StubRealtime()
    module = types.ModuleType("src.shared.python.realtime")
    module.register_channel = stub.register_channel  # type: ignore[attr-defined]
    module.subscribe = stub.subscribe  # type: ignore[attr-defined]
    module.publish = stub.publish  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"src.shared.python.realtime": module}, clear=False):
        yield stub


# --------------------------------------------------------------- construction


class TestConstruction:
    def test_records_channel(self) -> None:
        sub = TrainingJobLiveSubscriber("job-1")
        assert sub.job_id == "job-1"
        assert sub.channel == training_channel_for("job-1")
        assert sub.is_started is False

    def test_rejects_empty_job_id(self) -> None:
        with pytest.raises(ValueError):
            TrainingJobLiveSubscriber("")

    def test_rejects_non_callable_metric_callback(self) -> None:
        with pytest.raises(TypeError):
            TrainingJobLiveSubscriber(
                "job-1",
                on_metric="nope",  # type: ignore[arg-type]
            )

    def test_rejects_non_callable_status_callback(self) -> None:
        with pytest.raises(TypeError):
            TrainingJobLiveSubscriber(
                "job-1",
                on_status=42,  # type: ignore[arg-type]
            )


# ----------------------------------------------------------------- start/stop


class TestLifecycle:
    def test_start_subscribes_and_registers_channel(
        self, stub_realtime: _StubRealtime
    ) -> None:
        sub = TrainingJobLiveSubscriber("job-1")
        sub.start()
        try:
            assert sub.is_started is True
            assert stub_realtime.registered, "channel should be registered"
            assert stub_realtime.registered[0][0] == sub.channel
            assert len(stub_realtime.subscriptions[sub.channel]) == 1
        finally:
            sub.stop()

    def test_start_is_idempotent(self, stub_realtime: _StubRealtime) -> None:
        sub = TrainingJobLiveSubscriber("job-1")
        sub.start()
        sub.start()
        try:
            assert len(stub_realtime.subscriptions[sub.channel]) == 1
        finally:
            sub.stop()

    def test_stop_unsubscribes(self, stub_realtime: _StubRealtime) -> None:
        sub = TrainingJobLiveSubscriber("job-1")
        sub.start()
        sub.stop()
        assert sub.is_started is False
        assert stub_realtime.subscriptions.get(sub.channel, []) == []

    def test_stop_without_start_is_noop(self, stub_realtime: _StubRealtime) -> None:
        sub = TrainingJobLiveSubscriber("job-1")
        sub.stop()  # must not raise
        assert sub.is_started is False
        # No subscription ever registered:
        assert stub_realtime.subscriptions == {}


# --------------------------------------------------------------- dispatch path


class TestDispatch:
    def test_decodes_metric_payload(self, stub_realtime: _StubRealtime) -> None:
        seen: list[TrainingMetric] = []
        sub = TrainingJobLiveSubscriber("job-1", on_metric=seen.append)
        sub.start()
        try:
            metric = TrainingMetric(
                name="loss",
                value=0.5,
                step=3,
                timestamp=100.0,
                kind=MetricKind.LOSS,
            )
            stub_realtime.publish(
                sub.channel,
                {"event": "metric", "metric": training_metric_to_dict(metric)},
            )
            assert len(seen) == 1
            assert seen[0].name == "loss"
            assert seen[0].value == pytest.approx(0.5)
            assert seen[0].kind is MetricKind.LOSS
        finally:
            sub.stop()

    def test_decodes_status_payload_with_message(
        self, stub_realtime: _StubRealtime
    ) -> None:
        seen: list[tuple[TrainingStatus, str | None]] = []

        def on_status(status: TrainingStatus, message: str | None) -> None:
            seen.append((status, message))

        sub = TrainingJobLiveSubscriber("job-1", on_status=on_status)
        sub.start()
        try:
            stub_realtime.publish(
                sub.channel,
                {"event": "status", "status": "failed", "message": "boom"},
            )
            assert seen == [(TrainingStatus.FAILED, "boom")]
        finally:
            sub.stop()

    def test_decodes_status_payload_without_message(
        self, stub_realtime: _StubRealtime
    ) -> None:
        seen: list[tuple[TrainingStatus, str | None]] = []

        def on_status(status: TrainingStatus, message: str | None) -> None:
            seen.append((status, message))

        sub = TrainingJobLiveSubscriber("job-1", on_status=on_status)
        sub.start()
        try:
            stub_realtime.publish(sub.channel, {"event": "status", "status": "running"})
            assert seen == [(TrainingStatus.RUNNING, None)]
        finally:
            sub.stop()

    def test_drops_unknown_event(self, stub_realtime: _StubRealtime) -> None:
        on_metric = []
        on_status = []
        sub = TrainingJobLiveSubscriber(
            "job-1",
            on_metric=on_metric.append,
            on_status=lambda s, m: on_status.append((s, m)),
        )
        sub.start()
        try:
            stub_realtime.publish(sub.channel, {"event": "noise"})
            stub_realtime.publish(sub.channel, "not-a-dict")
            assert on_metric == []
            assert on_status == []
        finally:
            sub.stop()

    def test_drops_unknown_status_value(self, stub_realtime: _StubRealtime) -> None:
        seen: list[tuple[TrainingStatus, str | None]] = []
        sub = TrainingJobLiveSubscriber(
            "job-1", on_status=lambda s, m: seen.append((s, m))
        )
        sub.start()
        try:
            stub_realtime.publish(
                sub.channel,
                {"event": "status", "status": "made-up-status"},
            )
            assert seen == []
        finally:
            sub.stop()

    def test_drops_invalid_metric_payload(self, stub_realtime: _StubRealtime) -> None:
        seen: list[TrainingMetric] = []
        sub = TrainingJobLiveSubscriber("job-1", on_metric=seen.append)
        sub.start()
        try:
            # Missing required keys — decoder raises KeyError.
            stub_realtime.publish(
                sub.channel,
                {"event": "metric", "metric": {"name": "loss"}},
            )
            assert seen == []
        finally:
            sub.stop()

    def test_callback_exception_does_not_propagate(
        self, stub_realtime: _StubRealtime
    ) -> None:
        def boom(metric: TrainingMetric) -> None:
            raise RuntimeError("kaboom")

        sub = TrainingJobLiveSubscriber("job-1", on_metric=boom)
        sub.start()
        try:
            metric = TrainingMetric(name="loss", value=1.0, step=0, timestamp=0.0)
            # Should swallow the callback error so the transport thread
            # is not killed.
            stub_realtime.publish(
                sub.channel,
                {"event": "metric", "metric": training_metric_to_dict(metric)},
            )
        finally:
            sub.stop()

    def test_no_metric_callback_drops_metric_silently(
        self, stub_realtime: _StubRealtime
    ) -> None:
        sub = TrainingJobLiveSubscriber("job-1", on_status=lambda s, m: None)
        sub.start()
        try:
            metric = TrainingMetric(name="loss", value=0.0, step=0, timestamp=0.0)
            # Must not raise even though on_metric is None.
            stub_realtime.publish(
                sub.channel,
                {"event": "metric", "metric": training_metric_to_dict(metric)},
            )
        finally:
            sub.stop()

    def test_no_status_callback_drops_status_silently(
        self, stub_realtime: _StubRealtime
    ) -> None:
        sub = TrainingJobLiveSubscriber("job-1", on_metric=lambda m: None)
        sub.start()
        try:
            stub_realtime.publish(sub.channel, {"event": "status", "status": "running"})
        finally:
            sub.stop()
