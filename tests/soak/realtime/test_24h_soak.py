"""24-hour soak validation for the upstream-realtime Rust WS pub-sub.

Issue #5235 (follow-up to #5214). The acceptance band in #5214 is
"survives 24h subscribe-soak without thread/socket leak"; a 24-hour
test is not appropriate for the PR CI lane, so this module is gated:

* ``SOAK=1`` env var must be set for collection to proceed; otherwise
  every test in the module is skipped with a clear reason.
* The duration is configurable via ``SOAK_DURATION_SEC`` (default 60s,
  i.e. a fast smoke version usable from a dev machine). The nightly
  workflow sets it to 86400.
* The publish rate is configurable via ``SOAK_PUBLISH_HZ`` (default
  1000 Hz, matching the acceptance representative load in #5235).
* The subscriber count is configurable via ``SOAK_SUBSCRIBERS``
  (default 4, matching the #5235 spec).

What the test asserts (drift bounds calibrated from #5214 acceptance):

* RSS does not grow more than 10% per hour over the run window.
* The thread count does not grow more than ``+2`` over the run window.
* The open-FD count does not grow more than ``+8`` over the run window
  (small headroom for librt/jemalloc background threads on Linux).
* End-to-end one-hop latency stays under the #5214 budget
  (p50 < 10 ms, p99 < 50 ms) measured at 1 Hz across the window.

Sample series (timestamp, rss_mb, fds, threads, latency_ms) are written
to a CSV at ``SOAK_ARTIFACT_DIR/realtime_soak_samples.csv`` so that the
nightly workflow can attach them as build artifacts.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gating: SOAK=1 must be set, otherwise skip all tests in this module.
# ---------------------------------------------------------------------------
if os.environ.get("SOAK", "").strip() != "1":
    pytest.skip(
        "soak tests gated on SOAK=1 (long-running; nightly only)",
        allow_module_level=True,
    )

# upstream_realtime is the Rust wheel under test; without it the soak is
# meaningless. Skip cleanly if it's not installed.
pytest.importorskip("upstream_realtime")
pytest.importorskip("psutil")

import psutil  # noqa: E402  (must follow importorskip)

from src.shared.python.realtime.ws_pubsub import WSPubSub  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration knobs (all env-driven so the same test serves dev smoke,
# PR-CI smoke, and the 24h nightly).
# ---------------------------------------------------------------------------

DEFAULT_DURATION_SEC = 60
DEFAULT_PUBLISH_HZ = 1000.0
DEFAULT_SUBSCRIBERS = 4
DEFAULT_SAMPLE_INTERVAL_SEC = 1.0
# In smoke mode (short duration) we sample faster so we still get enough
# points for the per-window growth assertions to be statistically useful.
DEFAULT_SMOKE_SAMPLE_INTERVAL_SEC = 0.5

# Acceptance bands (from #5214 / #5235).
MAX_RSS_GROWTH_PER_HOUR_FRAC = 0.10  # 10 % per hour
MAX_THREAD_GROWTH = 2
MAX_FD_GROWTH = 8
LATENCY_P50_MS_BUDGET = 10.0
LATENCY_P99_MS_BUDGET = 50.0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"env {name} must be an int, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"env {name} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class SoakConfig:
    duration_sec: int
    publish_hz: float
    subscribers: int
    sample_interval_sec: float
    artifact_dir: Path

    @classmethod
    def from_env(cls) -> SoakConfig:
        duration = _env_int("SOAK_DURATION_SEC", DEFAULT_DURATION_SEC)
        publish_hz = _env_float("SOAK_PUBLISH_HZ", DEFAULT_PUBLISH_HZ)
        subs = _env_int("SOAK_SUBSCRIBERS", DEFAULT_SUBSCRIBERS)
        # Sample interval: 1 s for the 24h nightly, 0.5 s for the smoke.
        default_interval = (
            DEFAULT_SAMPLE_INTERVAL_SEC
            if duration >= 3600
            else DEFAULT_SMOKE_SAMPLE_INTERVAL_SEC
        )
        sample_interval = _env_float("SOAK_SAMPLE_INTERVAL_SEC", default_interval)

        artifact_dir = Path(
            os.environ.get("SOAK_ARTIFACT_DIR", "soak_artifacts")
        ).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if duration <= 0:
            raise ValueError("SOAK_DURATION_SEC must be > 0")
        if publish_hz <= 0:
            raise ValueError("SOAK_PUBLISH_HZ must be > 0")
        if subs <= 0:
            raise ValueError("SOAK_SUBSCRIBERS must be > 0")
        if sample_interval <= 0:
            raise ValueError("SOAK_SAMPLE_INTERVAL_SEC must be > 0")

        return cls(
            duration_sec=duration,
            publish_hz=publish_hz,
            subscribers=subs,
            sample_interval_sec=sample_interval,
            artifact_dir=artifact_dir,
        )


@dataclass
class Sample:
    elapsed_sec: float
    rss_mb: float
    fd_count: int
    thread_count: int
    latency_ms: float
    messages_sent: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _open_fd_count(proc: psutil.Process) -> int:
    """Cross-platform open-FD count. Falls back to handle count on Windows."""
    try:
        return proc.num_fds()  # POSIX
    except (AttributeError, NotImplementedError):
        pass
    try:
        return proc.num_handles()  # Windows
    except (AttributeError, NotImplementedError):
        return -1


def _write_samples_csv(samples: list[Sample], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "elapsed_sec",
                "rss_mb",
                "fd_count",
                "thread_count",
                "latency_ms",
                "messages_sent",
            ]
        )
        for s in samples:
            writer.writerow(
                [
                    f"{s.elapsed_sec:.3f}",
                    f"{s.rss_mb:.3f}",
                    s.fd_count,
                    s.thread_count,
                    f"{s.latency_ms:.6f}",
                    s.messages_sent,
                ]
            )


def _write_summary_json(
    cfg: SoakConfig,
    samples: list[Sample],
    latencies_ms: list[float],
    path: Path,
) -> None:
    rss_series = [s.rss_mb for s in samples]
    fd_series = [s.fd_count for s in samples if s.fd_count >= 0]
    thread_series = [s.thread_count for s in samples]
    summary = {
        "config": {
            "duration_sec": cfg.duration_sec,
            "publish_hz": cfg.publish_hz,
            "subscribers": cfg.subscribers,
            "sample_interval_sec": cfg.sample_interval_sec,
        },
        "samples": len(samples),
        "rss_mb": {
            "first": rss_series[0] if rss_series else None,
            "last": rss_series[-1] if rss_series else None,
            "max": max(rss_series) if rss_series else None,
            "min": min(rss_series) if rss_series else None,
        },
        "fd_count": {
            "first": fd_series[0] if fd_series else None,
            "last": fd_series[-1] if fd_series else None,
            "max": max(fd_series) if fd_series else None,
        },
        "thread_count": {
            "first": thread_series[0] if thread_series else None,
            "last": thread_series[-1] if thread_series else None,
            "max": max(thread_series) if thread_series else None,
        },
        "latency_ms": {
            "n": len(latencies_ms),
            "p50": _percentile(latencies_ms, 0.50) if latencies_ms else None,
            "p99": _percentile(latencies_ms, 0.99) if latencies_ms else None,
            "max": max(latencies_ms) if latencies_ms else None,
        },
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_realtime_ws_soak() -> None:
    """End-to-end soak: publisher + N subscribers, watch RSS/FDs/threads/latency.

    This test is gated on ``SOAK=1`` (see module-level skip). The default
    duration is 60 s for fast smoke runs; the nightly workflow sets
    ``SOAK_DURATION_SEC=86400`` for the full 24-hour validation.
    """
    cfg = SoakConfig.from_env()
    proc = psutil.Process(os.getpid())

    # OS-assigned port avoids collisions when multiple soak shards are
    # somehow scheduled on the same host.
    ps = WSPubSub(host="127.0.0.1", port=0, autostart=True, backend="rust")
    assert ps.backend == "rust", "soak requires the rust backend"

    received_counts = [0] * cfg.subscribers
    last_recv_ns = [0] * cfg.subscribers
    subscription_handles: list = []

    def _make_callback(idx: int):
        def _cb(_msg: dict) -> None:
            received_counts[idx] += 1
            last_recv_ns[idx] = time.perf_counter_ns()

        return _cb

    samples: list[Sample] = []
    latencies_ms: list[float] = []
    messages_sent = 0
    stop_event = threading.Event()
    publish_exc: list[BaseException] = []

    def _publish_loop() -> None:
        nonlocal messages_sent
        period = 1.0 / cfg.publish_hz
        next_t = time.perf_counter()
        try:
            while not stop_event.is_set():
                next_t += period
                ps.publish(
                    "pose/canonical",
                    {"seq": messages_sent, "ts_ns": time.perf_counter_ns()},
                )
                messages_sent += 1
                sleep_for = next_t - time.perf_counter()
                if sleep_for > 0 and stop_event.wait(sleep_for):
                    return
                # If sleep_for <= 0 we're behind schedule; skip the wait and
                # immediately publish the next one. This is fine for soak
                # purposes — we care about steady-state behavior, not strict
                # cadence.
        except BaseException as exc:  # noqa: BLE001
            publish_exc.append(exc)

    try:
        # Subscribe N consumers. We measure latency on subscriber 0 by
        # piggy-backing on its callback (cheap, no extra channel needed).
        sub0_latencies: list[float] = []

        def _cb_with_latency(msg: dict) -> None:
            recv_ns = time.perf_counter_ns()
            received_counts[0] += 1
            last_recv_ns[0] = recv_ns
            sent_ns = msg.get("ts_ns")
            if isinstance(sent_ns, int):
                sub0_latencies.append((recv_ns - sent_ns) / 1_000_000.0)

        subscription_handles.append(ps.subscribe("pose/canonical", _cb_with_latency))
        for i in range(1, cfg.subscribers):
            subscription_handles.append(
                ps.subscribe("pose/canonical", _make_callback(i))
            )

        # Let the broadcast subscribers park before the first publish.
        time.sleep(0.2)

        # Capture baseline metrics BEFORE starting the publisher so RSS
        # growth is measured against a steady-state floor that already
        # includes subscriber threads.
        baseline_rss_mb = proc.memory_info().rss / (1024 * 1024)
        baseline_fds = _open_fd_count(proc)
        baseline_threads = proc.num_threads()
        logger.info(
            "soak baseline: rss=%.1fMB fds=%d threads=%d duration=%ds "
            "publish_hz=%.0f subscribers=%d",
            baseline_rss_mb,
            baseline_fds,
            baseline_threads,
            cfg.duration_sec,
            cfg.publish_hz,
            cfg.subscribers,
        )

        pub_thread = threading.Thread(
            target=_publish_loop, name="soak-publisher", daemon=True
        )
        pub_thread.start()

        # Sample loop.
        start = time.perf_counter()
        next_sample = start
        deadline = start + cfg.duration_sec
        while True:
            now = time.perf_counter()
            if now >= deadline:
                break
            if now >= next_sample:
                rss_mb = proc.memory_info().rss / (1024 * 1024)
                fds = _open_fd_count(proc)
                threads = proc.num_threads()
                # Sample the most recent latency observation (if any) for
                # the CSV; full latency stats come from sub0_latencies.
                latency_ms = sub0_latencies[-1] if sub0_latencies else float("nan")
                samples.append(
                    Sample(
                        elapsed_sec=now - start,
                        rss_mb=rss_mb,
                        fd_count=fds,
                        thread_count=threads,
                        latency_ms=latency_ms,
                        messages_sent=messages_sent,
                    )
                )
                next_sample += cfg.sample_interval_sec
            # Pace the sampler against wallclock without hot-spinning.
            time.sleep(min(0.05, max(0.0, next_sample - time.perf_counter())))

        stop_event.set()
        pub_thread.join(timeout=10.0)
        if pub_thread.is_alive():
            pytest.fail("publish thread did not exit within 10s of stop signal")
        if publish_exc:
            raise publish_exc[0]

        latencies_ms = list(sub0_latencies)
    finally:
        # Always write artifacts so the nightly workflow can attach them
        # even on failure.
        cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cfg.artifact_dir / "realtime_soak_samples.csv"
        summary_path = cfg.artifact_dir / "realtime_soak_summary.json"
        try:
            _write_samples_csv(samples, csv_path)
            _write_summary_json(cfg, samples, latencies_ms, summary_path)
            logger.info("soak artifacts: %s, %s", csv_path, summary_path)
        except Exception:
            logger.exception("failed to write soak artifacts")

        # Tear down subscribers and server.
        for h in subscription_handles:
            try:
                h.unsubscribe()
            except Exception:
                logger.exception("subscriber teardown failed")
        try:
            ps.stop()
        except Exception:
            logger.exception("WSPubSub.stop() failed")

    # ----- assertions ------------------------------------------------------
    assert samples, "no samples captured (sample loop never executed)"
    assert messages_sent > 0, "publisher sent no messages"

    # Every subscriber should have received roughly the same number of
    # messages. Don't require equality (broadcast channels have transient
    # lag) but require each subscriber to be within 1% of the publisher
    # count across the full window.
    for i, n in enumerate(received_counts):
        assert n >= int(messages_sent * 0.99), (
            f"subscriber {i} received {n} / {messages_sent} "
            f"({n / max(messages_sent, 1):.4f}) — possible drop or stall"
        )

    # RSS growth bound: max-over-window / first-sample, normalized to a
    # per-hour rate so the same threshold serves the 60 s smoke and the
    # 24 h nightly.
    rss_series = [s.rss_mb for s in samples]
    rss_first = rss_series[0]
    rss_max = max(rss_series)
    rss_growth_frac = (rss_max - rss_first) / max(rss_first, 1e-9)
    hours = cfg.duration_sec / 3600.0
    # For windows shorter than 1 h we still demand the absolute growth
    # stays under 10 % of the baseline — i.e. we don't reward a short
    # window with a looser bound.
    rss_growth_per_hour = rss_growth_frac / hours if hours >= 1.0 else rss_growth_frac
    assert rss_growth_per_hour <= MAX_RSS_GROWTH_PER_HOUR_FRAC, (
        f"RSS grew {rss_growth_frac * 100:.2f}% over {cfg.duration_sec}s "
        f"({rss_growth_per_hour * 100:.2f}%/h, budget "
        f"{MAX_RSS_GROWTH_PER_HOUR_FRAC * 100:.0f}%/h): "
        f"first={rss_first:.1f}MB max={rss_max:.1f}MB"
    )

    # Thread count: must not grow more than MAX_THREAD_GROWTH over the
    # whole window. Use first sample as the post-warmup baseline.
    thread_series = [s.thread_count for s in samples]
    thread_first = thread_series[0]
    thread_max = max(thread_series)
    assert (thread_max - thread_first) <= MAX_THREAD_GROWTH, (
        f"thread count grew from {thread_first} to {thread_max} "
        f"(>{MAX_THREAD_GROWTH}); possible thread leak"
    )

    # FD count: similar bound; skip on platforms where _open_fd_count
    # returned -1 throughout.
    fd_series = [s.fd_count for s in samples if s.fd_count >= 0]
    if fd_series:
        fd_first = fd_series[0]
        fd_max = max(fd_series)
        assert (fd_max - fd_first) <= MAX_FD_GROWTH, (
            f"open FD count grew from {fd_first} to {fd_max} "
            f"(>{MAX_FD_GROWTH}); possible socket leak"
        )
    else:
        logger.warning("FD count unavailable on this platform; skipping FD check")

    # Latency: at full 1 kHz publish rate the latency sample list is
    # large; subsample to keep the percentile call cheap on 24 h runs.
    if latencies_ms:
        # Drop the first ~100 samples as warmup.
        steady = latencies_ms[100:] if len(latencies_ms) > 200 else latencies_ms
        p50 = _percentile(steady, 0.50)
        p99 = _percentile(steady, 0.99)
        mean = statistics.fmean(steady)
        logger.info(
            "soak latency: n=%d p50=%.3fms p99=%.3fms mean=%.3fms",
            len(steady),
            p50,
            p99,
            mean,
        )
        assert p50 < LATENCY_P50_MS_BUDGET, (
            f"p50 latency {p50:.3f}ms exceeds {LATENCY_P50_MS_BUDGET}ms budget"
        )
        assert p99 < LATENCY_P99_MS_BUDGET, (
            f"p99 latency {p99:.3f}ms exceeds {LATENCY_P99_MS_BUDGET}ms budget"
        )
    else:
        pytest.fail("no latency samples captured (subscriber 0 did not receive)")

    # Surface a concise stdout summary so it shows up in CI logs.
    print(
        "\nrealtime soak: duration={d}s sent={n} subs={s} "
        "rss=({rf:.1f}->{rm:.1f}MB, +{rg:.2f}%/h) "
        "threads=({tf}->{tm}) fds=({ff}->{fm}) "
        "p50={p50:.3f}ms p99={p99:.3f}ms".format(
            d=cfg.duration_sec,
            n=messages_sent,
            s=cfg.subscribers,
            rf=rss_first,
            rm=rss_max,
            rg=rss_growth_per_hour * 100,
            tf=thread_first,
            tm=thread_max,
            ff=(fd_series[0] if fd_series else -1),
            fm=(fd_series[-1] if fd_series else -1),
            p50=(
                _percentile(latencies_ms[100:], 0.50)
                if len(latencies_ms) > 200
                else _percentile(latencies_ms, 0.50)
            ),
            p99=(
                _percentile(latencies_ms[100:], 0.99)
                if len(latencies_ms) > 200
                else _percentile(latencies_ms, 0.99)
            ),
        ),
        file=sys.stderr,
    )
