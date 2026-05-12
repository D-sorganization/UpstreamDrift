"""Rust ↔ Python backend parity tests for the realtime WS pub-sub.

These tests are skipped when the ``upstream_realtime`` wheel is not
importable. The Python (FastAPI) backend is exercised in
``test_ws_pubsub_inproc.py``; here we focus on behavioural parity between
the two backends for the public ``WSPubSub`` API.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.shared.python.realtime.ws_pubsub import WSPubSub

pytestmark = pytest.mark.unit


pytest.importorskip("upstream_realtime")


def _wait_until(predicate, timeout_s: float = 2.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


@pytest.fixture()
def rust_pubsub() -> WSPubSub:
    ps = WSPubSub(host="127.0.0.1", port=0, autostart=True, backend="rust")
    yield ps
    ps.stop()


def test_rust_backend_is_active(rust_pubsub: WSPubSub) -> None:
    assert rust_pubsub.backend == "rust"
    assert rust_pubsub.port > 0


def test_publish_then_subscribe_receives_payloads(rust_pubsub: WSPubSub) -> None:
    received: list[dict] = []
    sub = rust_pubsub.subscribe("pose/canonical", received.append)
    try:
        # Subscriber thread must have entered recv() before we publish.
        time.sleep(0.05)
        for i in range(100):
            rust_pubsub.publish("pose/canonical", {"frame": i})
        assert _wait_until(lambda: len(received) == 100, timeout_s=5.0)
        assert [m["frame"] for m in received] == list(range(100))
    finally:
        sub.unsubscribe()


def test_invalid_channel_rejected(rust_pubsub: WSPubSub) -> None:
    with pytest.raises(ValueError):
        rust_pubsub.publish("BAD/Name", {"x": 1})
    with pytest.raises(ValueError):
        rust_pubsub.subscribe("BAD/Name", lambda _d: None)


def test_subscribers_isolated_by_channel(rust_pubsub: WSPubSub) -> None:
    pose: list[dict] = []
    target: list[dict] = []
    sub_a = rust_pubsub.subscribe("pose/canonical", pose.append)
    sub_b = rust_pubsub.subscribe("target/active", target.append)
    try:
        time.sleep(0.05)
        rust_pubsub.publish("pose/canonical", {"y": 1})
        rust_pubsub.publish("target/active", {"z": 2})
        assert _wait_until(lambda: pose and target, timeout_s=2.0)
        assert pose == [{"y": 1}]
        assert target == [{"z": 2}]
    finally:
        sub_a.unsubscribe()
        sub_b.unsubscribe()


def test_unsubscribe_stops_callback(rust_pubsub: WSPubSub) -> None:
    received: list[dict] = []
    sub = rust_pubsub.subscribe("pose/canonical", received.append)
    time.sleep(0.05)
    rust_pubsub.publish("pose/canonical", {"n": 0})
    assert _wait_until(lambda: len(received) == 1, timeout_s=2.0)
    sub.unsubscribe()
    rust_pubsub.publish("pose/canonical", {"n": 1})
    # Give the system a moment; nothing should land in received.
    time.sleep(0.2)
    assert received == [{"n": 0}]


def test_concurrent_publishers(rust_pubsub: WSPubSub) -> None:
    received: list[int] = []
    lock = threading.Lock()

    def collect(msg: dict) -> None:
        with lock:
            received.append(msg["i"])

    sub = rust_pubsub.subscribe("pose/canonical", collect)
    try:
        time.sleep(0.05)

        def worker(start: int, count: int) -> None:
            for i in range(start, start + count):
                rust_pubsub.publish("pose/canonical", {"i": i})

        threads = [threading.Thread(target=worker, args=(k * 50, 50)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert _wait_until(lambda: len(received) == 200, timeout_s=5.0)
        assert sorted(received) == list(range(200))
    finally:
        sub.unsubscribe()
