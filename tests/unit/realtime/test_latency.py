"""Latency benchmarks for the Rust realtime WS pub-sub backend.

Acceptance (issue #5214): p50 < 10 ms, p99 < 50 ms one-hop in-process.

The test is marked ``benchmark`` so it can be excluded from the default
fast suite; CI runs it on the latency-sensitive path. The 24-hour soak
test is deferred to a follow-up issue — see PR description.
"""

from __future__ import annotations

import threading
import time

import pytest

from src.shared.python.realtime.ws_pubsub import WSPubSub

pytestmark = [pytest.mark.unit, pytest.mark.benchmark]


pytest.importorskip("upstream_realtime")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def test_rust_one_hop_latency_meets_acceptance() -> None:
    """Measure one-hop latency over 10k paced ping-pong cycles.

    Acceptance from #5214: p50 < 10 ms, p99 < 50 ms.

    Each cycle: publish a message tagged with ``perf_counter_ns``; the
    subscriber callback drains a per-message ``threading.Event`` so the
    publisher only sends the next message once the previous one round-
    trips. This measures real one-hop latency (publish → broadcast →
    Python callback) without coupling it to in-flight queue depth.
    """
    ps = WSPubSub(host="127.0.0.1", port=0, autostart=True, backend="rust")
    assert ps.backend == "rust", "rust backend not active"

    n_messages = 10_000
    latencies_ms: list[float] = []
    arrived = threading.Event()
    state = {"last_send_ns": 0}

    def on_msg(_msg: dict) -> None:
        recv_ns = time.perf_counter_ns()
        latency_ms = (recv_ns - state["last_send_ns"]) / 1_000_000.0
        latencies_ms.append(latency_ms)
        arrived.set()

    sub = ps.subscribe("pose/canonical", on_msg)
    try:
        # Warm the recv() thread so it is parked in broadcast::Receiver::recv.
        time.sleep(0.1)

        for i in range(n_messages):
            arrived.clear()
            state["last_send_ns"] = time.perf_counter_ns()
            ps.publish("pose/canonical", {"i": i})
            # Block until the subscriber callback has observed the message.
            if not arrived.wait(timeout=1.0):
                pytest.fail(f"message {i} not received within 1s")

        # Discard the first 100 samples as warm-up.
        steady = latencies_ms[100:]
        p50 = _percentile(steady, 0.50)
        p99 = _percentile(steady, 0.99)
        print(
            f"\nupstream-realtime latency: n={len(steady)} "
            f"p50={p50:.3f}ms p99={p99:.3f}ms "
            f"max={max(steady):.3f}ms"
        )
        # Acceptance from #5214.
        assert p50 < 10.0, f"p50 {p50:.3f}ms exceeds 10ms budget"
        assert p99 < 50.0, f"p99 {p99:.3f}ms exceeds 50ms budget"
    finally:
        sub.unsubscribe()
        ps.stop()
