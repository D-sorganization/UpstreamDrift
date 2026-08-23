"""Tests for the off-GUI-thread Sidekick readiness probe (issue #8939).

DbC contract under test: the blocking readiness probe never executes on the
thread that owns the launcher widgets, and a slow probe does not stall the
Qt event loop.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

from src.launchers.sidekick_readiness_worker import (  # noqa: E402
    SidekickReadinessProbeThread,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def qt_app() -> QCoreApplication:
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def test_probe_requires_callable() -> None:
    with pytest.raises(TypeError, match="probe must be callable"):
        SidekickReadinessProbeThread(
            probe="not-callable",  # type: ignore[arg-type]
            expected_instance_id=None,
        )


def test_probe_runs_off_gui_thread_and_does_not_block_event_loop(
    qt_app: QCoreApplication,
) -> None:
    """A slow probe must run on a worker thread while the loop keeps ticking."""
    gui_thread_id = threading.get_ident()
    probe_thread_ids: list[int] = []
    results: list[object] = []
    tick_count = 0

    def slow_probe(*, expected_instance_id: str | None = None) -> SimpleNamespace:
        probe_thread_ids.append(threading.get_ident())
        time.sleep(0.2)
        return SimpleNamespace(ready=True, detail=expected_instance_id)

    worker = SidekickReadinessProbeThread(
        probe=slow_probe,
        expected_instance_id="instance-1",
    )
    loop = QEventLoop()

    def on_result(result: object) -> None:
        results.append(result)
        loop.quit()

    worker.readiness_ready.connect(on_result)

    ticker = QTimer()
    ticker.setInterval(10)

    def on_tick() -> None:
        nonlocal tick_count
        tick_count += 1

    ticker.timeout.connect(on_tick)
    ticker.start()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(loop.quit)
    watchdog.start(5_000)

    worker.start()
    loop.exec()
    ticker.stop()
    worker.wait(5_000)

    assert results, "probe result never arrived (watchdog timeout)"
    # DbC: the blocking probe never ran on the GUI thread.
    assert probe_thread_ids and probe_thread_ids[0] != gui_thread_id
    # The GUI event loop kept ticking during the 200 ms blocking probe.
    assert tick_count >= 5
    # The expected instance identity is forwarded to the probe.
    assert getattr(results[0], "detail", None) == "instance-1"
