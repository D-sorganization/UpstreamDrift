"""Short soak smoke for the upstream-realtime Rust WS pub-sub.

This is the PR-CI counterpart of the SOAK=1-gated 24h test in
``tests/soak/realtime/test_24h_soak.py`` (issue #5235). It runs the
same publisher + N-subscriber topology for ~30 s and asserts the basic
"no obvious leak, latency in budget" properties, so PRs that regress
the realtime hot path get caught before the nightly window.

Budget: < 60 s wall-clock on a CI runner. Skipped cleanly when the Rust
wheel or psutil isn't available.
"""

from __future__ import annotations

import contextlib
import threading
import time

import pytest

pytest.importorskip("upstream_realtime")
pytest.importorskip("psutil")

import os  # noqa: E402

import psutil  # noqa: E402

from src.shared.python.realtime.ws_pubsub import WSPubSub  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.benchmark]


SMOKE_DURATION_SEC = 30.0
SMOKE_PUBLISH_HZ = 500.0  # half rate to keep CI runners happy
SMOKE_SUBSCRIBERS = 4
# Bounds are looser than the nightly because of CI variance, but a regression
# that doubles RSS or leaks tens of threads will still trip them.
SMOKE_MAX_RSS_GROWTH_FRAC = 0.50
SMOKE_MAX_THREAD_GROWTH = 4
SMOKE_MAX_FD_GROWTH = 16
SMOKE_LATENCY_P99_MS_BUDGET = 100.0  # 2x the strict 50ms band


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _fd_count(proc: psutil.Process) -> int:
    try:
        return proc.num_fds()
    except (AttributeError, NotImplementedError):
        pass
    try:
        return proc.num_handles()
    except (AttributeError, NotImplementedError):
        return -1


def test_realtime_ws_short_soak_smoke() -> None:
    """30-second smoke version of the 24h soak validation."""
    proc = psutil.Process(os.getpid())

    ps = WSPubSub(host="127.0.0.1", port=0, autostart=True, backend="rust")
    assert ps.backend == "rust", "smoke soak requires the rust backend"

    latencies_ms: list[float] = []
    received_counts = [0] * SMOKE_SUBSCRIBERS
    handles: list = []
    stop = threading.Event()
    sent = 0

    def _cb_with_latency(msg: dict) -> None:
        received_counts[0] += 1
        ts = msg.get("ts_ns")
        if isinstance(ts, int):
            latencies_ms.append((time.perf_counter_ns() - ts) / 1_000_000.0)

    def _make_cb(i: int):
        def _cb(_msg: dict) -> None:
            received_counts[i] += 1

        return _cb

    def _publish_loop() -> None:
        nonlocal sent
        period = 1.0 / SMOKE_PUBLISH_HZ
        next_t = time.perf_counter()
        while not stop.is_set():
            next_t += period
            ps.publish("pose/canonical", {"seq": sent, "ts_ns": time.perf_counter_ns()})
            sent += 1
            wait = next_t - time.perf_counter()
            if wait > 0 and stop.wait(wait):
                return

    try:
        handles.append(ps.subscribe("pose/canonical", _cb_with_latency))
        for i in range(1, SMOKE_SUBSCRIBERS):
            handles.append(ps.subscribe("pose/canonical", _make_cb(i)))

        time.sleep(0.2)  # let receivers park

        rss0 = proc.memory_info().rss / (1024 * 1024)
        fds0 = _fd_count(proc)
        threads0 = proc.num_threads()

        pub = threading.Thread(
            target=_publish_loop, name="smoke-publisher", daemon=True
        )
        pub.start()

        rss_peak = rss0
        threads_peak = threads0
        fds_peak = fds0
        end = time.perf_counter() + SMOKE_DURATION_SEC
        while time.perf_counter() < end:
            time.sleep(0.5)
            rss_peak = max(rss_peak, proc.memory_info().rss / (1024 * 1024))
            threads_peak = max(threads_peak, proc.num_threads())
            f = _fd_count(proc)
            if f >= 0:
                fds_peak = max(fds_peak, f)

        stop.set()
        pub.join(timeout=5.0)
        assert not pub.is_alive(), "smoke publisher did not exit on stop"
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.unsubscribe()
        with contextlib.suppress(Exception):
            ps.stop()

    assert sent > 0, "smoke publisher sent zero messages"
    for i, n in enumerate(received_counts):
        # Looser than the 24 h test — CI variance on shared runners is real.
        assert n >= int(sent * 0.90), (
            f"subscriber {i} received {n}/{sent} messages ({n / max(sent, 1):.3f})"
        )

    rss_growth = (rss_peak - rss0) / max(rss0, 1e-9)
    assert rss_growth <= SMOKE_MAX_RSS_GROWTH_FRAC, (
        f"smoke RSS grew {rss_growth * 100:.1f}% (>{SMOKE_MAX_RSS_GROWTH_FRAC * 100:.0f}%): "
        f"{rss0:.1f}MB -> {rss_peak:.1f}MB"
    )
    assert (threads_peak - threads0) <= SMOKE_MAX_THREAD_GROWTH, (
        f"thread count {threads0} -> {threads_peak} (>{SMOKE_MAX_THREAD_GROWTH})"
    )
    if fds0 >= 0:
        assert (fds_peak - fds0) <= SMOKE_MAX_FD_GROWTH, (
            f"open FD count {fds0} -> {fds_peak} (>{SMOKE_MAX_FD_GROWTH})"
        )

    if latencies_ms:
        steady = latencies_ms[50:] if len(latencies_ms) > 100 else latencies_ms
        p99 = _percentile(steady, 0.99)
        assert p99 < SMOKE_LATENCY_P99_MS_BUDGET, (
            f"smoke p99 latency {p99:.2f}ms exceeds {SMOKE_LATENCY_P99_MS_BUDGET}ms"
        )
    else:
        pytest.fail("no latency samples captured by smoke soak")
