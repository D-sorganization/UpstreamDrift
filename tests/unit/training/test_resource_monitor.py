"""Tests for :mod:`training.resource_monitor`.

Skipped cleanly when psutil isn't installed — the monitor is an
optional surface; tests must not fail the suite on hosts without it.
"""

from __future__ import annotations

import pytest

psutil = pytest.importorskip("psutil")

from training.resource_monitor import (  # noqa: E402
    GpuSample,
    ResourceMonitor,
    ResourceSample,
)

pytestmark = pytest.mark.unit


class TestGpuSample:
    def test_construct(self) -> None:
        s = GpuSample(
            index=0,
            name="RTX A6000",
            utilization_percent=12.5,
            memory_used_mb=2000,
            memory_total_mb=48000,
        )
        assert s.index == 0
        assert s.name == "RTX A6000"

    def test_rejects_negative_index(self) -> None:
        with pytest.raises(ValueError):
            GpuSample(
                index=-1,
                name="x",
                utilization_percent=0.0,
                memory_used_mb=0,
                memory_total_mb=1,
            )

    def test_rejects_oversubscribed_memory(self) -> None:
        with pytest.raises(ValueError):
            GpuSample(
                index=0,
                name="x",
                utilization_percent=0.0,
                memory_used_mb=200,
                memory_total_mb=100,
            )

    def test_rejects_out_of_range_util(self) -> None:
        with pytest.raises(ValueError):
            GpuSample(
                index=0,
                name="x",
                utilization_percent=120.0,
                memory_used_mb=0,
                memory_total_mb=1,
            )


class TestResourceSample:
    def test_construct(self) -> None:
        s = ResourceSample(
            timestamp=100.0,
            cpu_percent=25.0,
            memory_used_mb=1024,
            memory_total_mb=8192,
        )
        assert s.memory_percent == pytest.approx(12.5)

    def test_rejects_zero_total_memory(self) -> None:
        with pytest.raises(ValueError):
            ResourceSample(
                timestamp=0.0,
                cpu_percent=10.0,
                memory_used_mb=0,
                memory_total_mb=0,
            )


class TestResourceMonitor:
    def test_sample_once_returns_sample(self) -> None:
        monitor = ResourceMonitor()
        sample = monitor.sample_once()
        assert isinstance(sample, ResourceSample)
        assert 0.0 <= sample.cpu_percent <= 100.0
        assert sample.memory_total_mb > 0

    def test_latest_updated_after_sample(self) -> None:
        monitor = ResourceMonitor()
        assert monitor.latest is None
        monitor.sample_once()
        assert monitor.latest is not None

    def test_on_sample_callback_invoked(self) -> None:
        received: list[ResourceSample] = []
        monitor = ResourceMonitor(on_sample=received.append)
        monitor.sample_once()
        assert len(received) == 1

    def test_callback_exception_logged_not_raised(self) -> None:
        def bad(_s: ResourceSample) -> None:
            raise RuntimeError("oops")

        monitor = ResourceMonitor(on_sample=bad)
        monitor.sample_once()  # must not raise

    def test_rejects_fast_interval(self) -> None:
        with pytest.raises(ValueError):
            ResourceMonitor(sample_interval_s=0.1)

    def test_start_stop(self) -> None:
        monitor = ResourceMonitor(sample_interval_s=0.25)
        monitor.start()
        try:
            # Wait briefly for at least one sample.
            import time as _time

            deadline = _time.time() + 1.5
            while _time.time() < deadline and monitor.latest is None:
                _time.sleep(0.05)
            assert monitor.latest is not None
        finally:
            monitor.stop()

    def test_start_is_idempotent(self) -> None:
        monitor = ResourceMonitor(sample_interval_s=0.25)
        try:
            monitor.start()
            monitor.start()  # no error
        finally:
            monitor.stop()
