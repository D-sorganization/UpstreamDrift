"""Lifecycle/concurrency/health tests for the realtime facade.

Covers issue #7148 (single-transport invariant + shutdown) and #7149 D1/D2
(publish health latching + typed publish error).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.shared.python.realtime import api
from src.shared.python.realtime import transport_file

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_realtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REALTIME_FILE_ROOT", str(tmp_path))
    api.shutdown_realtime()
    yield
    api.shutdown_realtime()


def test_get_transport_constructs_exactly_one_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construct_count = 0
    real_init = transport_file.FileTransport.__init__

    def _counting_init(self, *args, **kwargs):
        nonlocal construct_count
        construct_count += 1
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(transport_file.FileTransport, "__init__", _counting_init)

    barrier = threading.Barrier(10)
    results = []

    def _worker() -> None:
        barrier.wait()
        results.append(api._get_transport())

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construct_count == 1
    assert len({id(r) for r in results}) == 1


def test_shutdown_realtime_is_idempotent_and_clears_transport() -> None:
    api.publish("pose/canonical", {"x": 1})
    assert api._TRANSPORT is not None
    api.shutdown_realtime()
    assert api._TRANSPORT is None
    # Second call must not raise.
    api.shutdown_realtime()
    assert api._TRANSPORT is None


def test_shutdown_stops_callbacks_across_tests() -> None:
    received: list = []
    sub = api.subscribe("pose/canonical", received.append)
    api.shutdown_realtime()
    # After shutdown the old subscription's transport is gone; a fresh publish
    # on a new transport must not reach the old callback.
    api.publish("pose/canonical", {"y": 2})
    assert received == []
    sub.unsubscribe()


def test_publish_failure_latches_health_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom(self, channel, payload):
        raise transport_file.RealtimePublishError("disk full")

    monkeypatch.setattr(transport_file.FileTransport, "publish", _boom)

    assert api.is_healthy() is True
    api.publish("pose/canonical", {"a": 1})
    api.publish("pose/canonical", {"a": 2})

    assert api.is_healthy() is False
    assert "disk full" in (api.validate_realtime() or "")
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1  # warn once, not per call


def test_publish_recovers_clears_health(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real_publish = transport_file.FileTransport.publish

    def _flaky(self, channel, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise transport_file.RealtimePublishError("transient")
        return real_publish(self, channel, payload)

    monkeypatch.setattr(transport_file.FileTransport, "publish", _flaky)

    api.publish("pose/canonical", {"a": 1})
    assert api.is_healthy() is False
    api.publish("pose/canonical", {"a": 2})
    assert api.is_healthy() is True


def test_transport_publish_raises_typed_error_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = transport_file.FileTransport(transport_file.default_channel_path)
    real_open = Path.open

    def _maybe_bad_open(self, mode="r", *a, **k):
        if "a" in mode:
            raise OSError("no space left on device")
        return real_open(self, mode, *a, **k)

    monkeypatch.setattr(Path, "open", _maybe_bad_open)
    with pytest.raises(transport_file.RealtimePublishError):
        transport.publish("pose/canonical", {"a": 1})
