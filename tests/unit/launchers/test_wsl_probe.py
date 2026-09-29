"""Non-blocking WSL probe tests (#8903).

The WSL-mode checkbox used to run ``subprocess.run(["wsl", ...])`` on the GUI
thread (up to 5 s freeze). The probe now runs in
:class:`src.launchers.wsl_probe.WslAvailabilityWorker` and lands via signal.
"""

from __future__ import annotations

import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.launchers import wsl_probe  # noqa: E402
from src.launchers.wsl_probe import (  # noqa: E402
    WslAvailabilityWorker,
    WslProbeResult,
    probe_wsl_available,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    wsl_probe.reset_wsl_probe_cache()
    yield
    wsl_probe.reset_wsl_probe_cache()


def _completed(returncode: int, stdout: bytes) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout)


class TestProbeFunction:
    def test_available_when_ubuntu_listed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            wsl_probe.subprocess,
            "run",
            lambda *a, **k: _completed(0, "Ubuntu-22.04\n".encode("utf-16-le")),
        )
        result = probe_wsl_available()
        assert result.available is True
        assert result.detail == ""

    def test_unavailable_when_no_ubuntu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            wsl_probe.subprocess,
            "run",
            lambda *a, **k: _completed(0, "Debian\n".encode("utf-16-le")),
        )
        result = probe_wsl_available()
        assert result.available is False
        assert "Ubuntu" in result.detail

    def test_unavailable_on_nonzero_returncode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            wsl_probe.subprocess, "run", lambda *a, **k: _completed(1, b"")
        )
        assert probe_wsl_available().available is False

    def test_unavailable_when_wsl_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*a: Any, **k: Any) -> None:
            raise OSError("wsl.exe not found")

        monkeypatch.setattr(wsl_probe.subprocess, "run", _raise)
        result = probe_wsl_available()
        assert result.available is False
        assert "wsl.exe not found" in result.detail

    def test_unavailable_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*a: Any, **k: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="wsl", timeout=5)

        monkeypatch.setattr(wsl_probe.subprocess, "run", _raise)
        assert probe_wsl_available().available is False

    def test_rejects_nonpositive_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            probe_wsl_available(timeout=0)


class TestResultCache:
    def test_round_trip(self) -> None:
        assert wsl_probe.cached_wsl_result() is None
        result = WslProbeResult(available=True)
        wsl_probe.store_wsl_result(result)
        assert wsl_probe.cached_wsl_result() is result
        wsl_probe.reset_wsl_probe_cache()
        assert wsl_probe.cached_wsl_result() is None

    def test_store_rejects_none(self) -> None:
        with pytest.raises(ValueError, match="result"):
            wsl_probe.store_wsl_result(None)  # type: ignore[arg-type]


class TestWorkerRunsOffGuiThread:
    def test_probe_executes_on_worker_thread(self, qapp: QApplication) -> None:
        main_thread = threading.get_ident()
        probe_thread: list[int] = []
        received: list[WslProbeResult] = []

        def _probe() -> WslProbeResult:
            probe_thread.append(threading.get_ident())
            return WslProbeResult(available=True)

        worker = WslAvailabilityWorker(probe=_probe)
        worker.result_ready.connect(received.append)
        worker.start()
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        worker.wait(1000)

        assert received and received[0].available is True
        assert probe_thread and probe_thread[0] != main_thread

    def test_slow_probe_does_not_block_event_loop(self, qapp: QApplication) -> None:
        """A 1 s probe must leave the GUI event loop responsive throughout."""
        received: list[WslProbeResult] = []

        def _slow_probe() -> WslProbeResult:
            time.sleep(1.0)
            return WslProbeResult(available=False, detail="slow")

        worker = WslAvailabilityWorker(probe=_slow_probe)
        worker.result_ready.connect(received.append)
        start = time.monotonic()
        worker.start()

        # The GUI thread keeps pumping events while the probe sleeps; every
        # iteration must return quickly (would stall >=1 s if the probe ran
        # on this thread).
        iterations = 0
        max_pump = 0.0
        while not received and time.monotonic() - start < 5.0:
            t0 = time.monotonic()
            qapp.processEvents()
            max_pump = max(max_pump, time.monotonic() - t0)
            iterations += 1
            time.sleep(0.005)
        worker.wait(1000)

        assert received, "probe result never arrived via signal"
        assert iterations > 10, "event loop starved while probe ran"
        assert max_pump < 0.5, f"processEvents blocked for {max_pump:.2f}s"

    def test_probe_exception_reports_unavailable(self, qapp: QApplication) -> None:
        received: list[WslProbeResult] = []

        def _broken_probe() -> WslProbeResult:
            raise RuntimeError("probe exploded")

        worker = WslAvailabilityWorker(probe=_broken_probe)
        worker.result_ready.connect(received.append)
        worker.start()
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        worker.wait(1000)

        assert received and received[0].available is False
        assert "probe exploded" in received[0].detail


class TestCheckboxHandlerNonBlocking:
    """The stateChanged handler must return immediately (direct call removed)."""

    def test_handler_returns_immediately_with_slow_probe(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PyQt6.QtWidgets import QCheckBox, QWidget

        from src.launchers.launcher_dialogs import DialogsManager

        def _slow_probe() -> WslProbeResult:
            time.sleep(2.0)
            return WslProbeResult(available=False, detail="slow WSL")

        monkeypatch.setattr(wsl_probe, "probe_wsl_available", _slow_probe)

        launcher = QWidget()
        launcher.loading = False
        launcher.toast_manager = None
        launcher.chk_wsl = QCheckBox(launcher)
        launcher.chk_docker = QCheckBox(launcher)
        launcher.chk_windows = QCheckBox(launcher)
        launcher.chk_wsl.setChecked(True)
        manager = DialogsManager(launcher)
        monkeypatch.setattr(
            DialogsManager, "update_execution_status", lambda self: None
        )
        monkeypatch.setattr(
            DialogsManager,
            "_warn_wsl_unavailable",
            lambda self, error: launcher.chk_wsl.setChecked(False),
        )

        start = time.monotonic()
        manager._on_wsl_mode_changed(2)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, f"handler blocked GUI thread for {elapsed:.2f}s"
        # In-progress state: checkbox disabled while the probe runs.
        assert launcher.chk_wsl.isEnabled() is False

        # Result lands via signal on the GUI thread.
        deadline = time.monotonic() + 6.0
        while manager._wsl_probe_worker is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert manager._wsl_probe_worker is None, "probe result never applied"
        assert launcher.chk_wsl.isEnabled() is True
        assert launcher.chk_wsl.isChecked() is False  # reverted: unavailable
        assert wsl_probe.cached_wsl_result() is not None

    def test_cached_result_skips_probe(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PyQt6.QtWidgets import QCheckBox, QWidget

        from src.launchers.launcher_dialogs import DialogsManager

        def _boom() -> WslProbeResult:
            raise AssertionError("probe must not run when cached")

        monkeypatch.setattr(wsl_probe, "probe_wsl_available", _boom)
        wsl_probe.store_wsl_result(WslProbeResult(available=True))

        launcher = QWidget()
        launcher.loading = False
        launcher.toast_manager = None
        launcher.chk_wsl = QCheckBox(launcher)
        launcher.chk_docker = QCheckBox(launcher)
        launcher.chk_windows = QCheckBox(launcher)
        launcher.chk_wsl.setChecked(True)
        manager = DialogsManager(launcher)
        monkeypatch.setattr(
            DialogsManager, "update_execution_status", lambda self: None
        )

        manager._on_wsl_mode_changed(2)
        assert launcher.chk_wsl.isEnabled() is True
        assert launcher.chk_wsl.isChecked() is True
