"""Tests for the bounded splash lifetime and startup session (issue #8360).

Covers the Qt-facing half of the fix:

* ``AsyncStartupWorker`` records named, timestamped phases, degrades on
  optional-provider failure and only fails hard on the registry.
* ``SplashScreen`` ignores late callbacks once its C++ object is deleted.
* ``StartupSession`` bounds the splash lifetime with a watchdog, opens the
  shell in degraded mode, ignores stale worker generations after a retry,
  and never quits the application on an optional-provider failure.
* ``StartupFailureDialog`` exposes deterministic, keyboard-accessible
  Retry / Continue / Copy diagnostics / Close actions.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from src.launchers.startup import (
    PHASE_DOCKER,
    PHASE_ENGINES,
    PHASE_REGISTRY,
    PHASE_TOOLS_PROVIDER,
    AsyncStartupWorker,
    SplashScreen,
    StartupResults,
)
from src.launchers.startup_failure_dialog import (
    StartupFailureAction,
    StartupFailureDialog,
)
from src.launchers.startup_phases import (
    CATEGORY_MISSING_CHECKOUT,
    CATEGORY_TIMEOUT,
    OUTCOME_DEGRADED,
    OUTCOME_OK,
    OUTCOME_TIMEOUT,
    ProviderProbeResult,
    StartupPhaseRecord,
)
from src.launchers.startup_session import StartupSession, is_alive

pytestmark = pytest.mark.unit


def _spin(ms: int) -> None:
    """Pump the Qt event loop for ``ms`` milliseconds."""
    QTest.qWait(ms)


def _wait_until(predicate, timeout_ms: int = 3000) -> bool:
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.start(timeout_ms)
    while not predicate():
        if not deadline.isActive():
            return False
        _spin(5)
    return True


class _FakeWorker(QObject):
    """Deterministic stand-in for ``AsyncStartupWorker``.

    ``behaviour`` is one of ``"success"``, ``"degraded"``, ``"error"`` or
    ``"hang"``; signals are emitted from a zero-delay timer so they arrive
    through the event loop exactly like queued cross-thread signals.
    """

    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, behaviour: str) -> None:
        super().__init__()
        self.behaviour = behaviour
        self.results = StartupResults()
        self.results.registry = MagicMock(name="registry")
        self.timeline = MagicMock()
        self.timeline.records = ()
        self.timeline.elapsed_ms.return_value = 7
        self.started = False
        self.interrupted = False
        self._running = False

    # QThread-compatible surface used by the session ---------------------------
    def start(self) -> None:
        self.started = True
        self._running = True
        QTimer.singleShot(0, self._run)

    def isRunning(self) -> bool:  # noqa: N802 - QThread API
        return self._running

    def wait(self, _ms: int) -> bool:
        return not self._running

    def requestInterruption(self) -> None:  # noqa: N802 - QThread API
        self.interrupted = True

    def diagnostics_text(self) -> str:
        return f"diagnostics for {self.behaviour}"

    def _run(self) -> None:
        self.progress_signal.emit(f"{PHASE_REGISTRY}...", 10)
        if self.behaviour == "hang":
            return  # never reports back; the watchdog must rescue the splash
        if self.behaviour == "error":
            self._running = False
            self.error_signal.emit("Loading model registry failed [import_failure]")
            self.finished.emit()
            return
        if self.behaviour == "degraded":
            self.results.phases = (
                StartupPhaseRecord(PHASE_REGISTRY, "t", 1, OUTCOME_OK),
                StartupPhaseRecord(
                    PHASE_TOOLS_PROVIDER,
                    "t",
                    1,
                    OUTCOME_DEGRADED,
                    CATEGORY_MISSING_CHECKOUT,
                    "no Tools",
                ),
            )
        self._running = False
        self.finished_signal.emit(self.results)
        self.finished.emit()


class _FakeShell(QWidget):
    """Minimal shell exposing what ``StartupSession`` touches."""

    def __init__(self) -> None:
        super().__init__()
        self.loading = True
        self.applied: list[StartupResults] = []
        self.toasts: list[tuple[str, str]] = []
        self.fail_apply = False

    def update_startup_results(self, results: StartupResults) -> None:
        if self.fail_apply:
            raise RuntimeError("layout exploded")
        self.loading = False
        self.applied.append(results)

    def show_toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _FakeDialog:
    """Records the failure presentation and returns a scripted action."""

    calls: list[tuple[str, str]] = []
    action = StartupFailureAction.CONTINUE

    def __init__(self, message: str, diagnostics: str, *, parent=None) -> None:
        type(self).calls.append((message, diagnostics))

    def ask(self) -> StartupFailureAction:
        return type(self).action


@pytest.fixture
def shell(qapp) -> Iterator[_FakeShell]:
    widget = _FakeShell()
    yield widget
    widget.close()
    widget.deleteLater()
    _spin(1)


@pytest.fixture
def splash(qapp) -> Iterator[SplashScreen]:
    with patch("src.launchers.startup.QApplication.processEvents"):
        screen = SplashScreen()
        screen.show()
        yield screen
        if is_alive(screen):
            screen.close()
            screen.deleteLater()
            _spin(1)


@pytest.fixture(autouse=True)
def _reset_dialog() -> Iterator[None]:
    _FakeDialog.calls = []
    _FakeDialog.action = StartupFailureAction.CONTINUE
    yield


def _session(splash, shell, behaviour: str, **kwargs) -> StartupSession:
    workers: list[_FakeWorker] = []

    def factory() -> _FakeWorker:
        worker = _FakeWorker(behaviour)
        workers.append(worker)
        return worker

    session = StartupSession(
        splash=splash,
        shell=shell,
        worker_factory=factory,  # type: ignore[arg-type]
        dialog_factory=_FakeDialog,  # type: ignore[arg-type]
        quit_callback=kwargs.pop("quit_callback", MagicMock()),
        **kwargs,
    )
    session._test_workers = workers  # type: ignore[attr-defined]
    return session


# ---------------------------------------------------------------------------
# StartupSession
# ---------------------------------------------------------------------------


def test_successful_startup_opens_shell_and_finishes_splash(splash, shell) -> None:
    session = _session(splash, shell, "success")
    session.start()
    assert _wait_until(lambda: session.settled)
    assert shell.applied and shell.loading is False
    assert not splash.isVisible()
    assert shell.toasts == []
    assert _FakeDialog.calls == []


def test_optional_provider_failure_opens_shell_degraded(splash, shell) -> None:
    """A missing Tools/Rate provider must never keep the shell from opening."""
    session = _session(splash, shell, "degraded")
    session.start()
    assert _wait_until(lambda: session.settled)
    assert shell.applied and shell.loading is False
    assert not splash.isVisible()
    assert _FakeDialog.calls == [], "optional degradation is not a hard failure"
    (message, kind) = shell.toasts[0]
    assert kind == "warning"
    assert PHASE_TOOLS_PROVIDER in message
    assert CATEGORY_MISSING_CHECKOUT in message


def test_hard_failure_presents_dialog_and_continue_opens_shell(splash, shell) -> None:
    quit_cb = MagicMock()
    session = _session(splash, shell, "error", quit_callback=quit_cb)
    session.start()
    assert _wait_until(lambda: session.settled)
    assert len(_FakeDialog.calls) == 1
    message, diagnostics = _FakeDialog.calls[0]
    assert "import_failure" in message
    assert diagnostics == "diagnostics for error"
    # "Continue without provider" opens the shell in degraded mode.
    assert shell.applied and shell.loading is False
    assert not splash.isVisible()
    assert any("degraded mode" in text for text, _ in shell.toasts)
    quit_cb.assert_not_called()


def test_close_action_quits_after_shutdown(splash, shell) -> None:
    _FakeDialog.action = StartupFailureAction.CLOSE
    quit_cb = MagicMock()
    session = _session(splash, shell, "error", quit_callback=quit_cb)
    session.start()
    assert _wait_until(lambda: session.settled)
    quit_cb.assert_called_once()
    assert shell.applied == []


def test_watchdog_bounds_splash_lifetime_when_worker_never_reports(
    splash, shell
) -> None:
    """A never-completing worker cannot hold the splash past ``timeout_s``."""
    session = _session(splash, shell, "hang", timeout_s=0.1)
    session.start()
    assert _wait_until(lambda: session.settled, timeout_ms=2000)
    assert len(_FakeDialog.calls) == 1
    message, _ = _FakeDialog.calls[0]
    assert "did not complete within 0.1s" in message
    assert PHASE_REGISTRY in message, "diagnostics name the stalled phase"
    # Continue-without-provider still opens the shell.
    assert shell.loading is False
    assert not splash.isVisible()


def test_retry_starts_new_generation_and_ignores_stale_worker(splash, shell) -> None:
    """After Retry, callbacks from the old (hung) worker are ignored."""
    _FakeDialog.action = StartupFailureAction.RETRY
    session = _session(splash, shell, "hang", timeout_s=0.1)
    session.start()
    assert _wait_until(lambda: session.generation == 2, timeout_ms=2000)
    workers = session._test_workers  # type: ignore[attr-defined]
    assert len(workers) == 2
    assert workers[0].interrupted is True
    # Retry re-shows the splash and re-arms the watchdog for the new worker.
    assert splash.isVisible()
    # A late signal from the retired generation must not touch the shell.
    workers[0].finished_signal.emit(workers[0].results)
    _spin(5)
    assert shell.applied == []
    # Stop the retry loop: the second watchdog fires, Close settles it.
    _FakeDialog.action = StartupFailureAction.CLOSE
    assert _wait_until(lambda: session.settled, timeout_ms=2000)
    assert session.generation == 2
    assert len(_FakeDialog.calls) == 2


def test_late_callbacks_skip_deleted_splash(qapp, shell) -> None:
    """Progress arriving after the splash was deleted must not raise."""
    with patch("src.launchers.startup.QApplication.processEvents"):
        screen = SplashScreen()
    session = _session(screen, shell, "hang", timeout_s=5)
    session.start()
    screen.deleteLater()
    _spin(5)
    assert not is_alive(screen)
    # Direct call path: the splash itself guards against deletion.
    screen.show_message("late", 50)
    # Session path: routed callback is dropped without touching Qt.
    session._on_progress(session.generation, "late", 60)
    session.shutdown()


def test_apply_failure_still_dismisses_splash_in_degraded_mode(splash, shell) -> None:
    shell.fail_apply = True
    session = _session(splash, shell, "success")
    session.start()
    assert _wait_until(lambda: session.settled)
    assert shell.loading is False
    assert not splash.isVisible()
    assert any(kind == "error" for _, kind in shell.toasts)


def test_session_validates_arguments(splash, shell) -> None:
    with pytest.raises(ValueError):
        StartupSession(splash=splash, shell=None, worker_factory=lambda: None)
    with pytest.raises(TypeError):
        StartupSession(splash=splash, shell=shell, worker_factory="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        StartupSession(
            splash=splash, shell=shell, worker_factory=lambda: None, timeout_s=0
        )


def test_is_alive_handles_none_and_plain_objects(qapp) -> None:
    assert is_alive(None) is False
    assert is_alive(object()) is True
    widget = QWidget()
    assert is_alive(widget) is True
    widget.deleteLater()
    _spin(1)
    assert is_alive(widget) is False


# ---------------------------------------------------------------------------
# AsyncStartupWorker phases
# ---------------------------------------------------------------------------


@pytest.fixture
def worker() -> AsyncStartupWorker:
    instance = AsyncStartupWorker(Path("fake_root"))
    instance.progress_signal = MagicMock()
    instance.finished_signal = MagicMock()
    instance.error_signal = MagicMock()
    return instance


def _run_worker(worker: AsyncStartupWorker, **phase_bodies) -> None:
    patches = [patch.object(worker, "msleep")]
    patches.extend(
        patch.object(worker, name, side_effect=body)
        for name, body in phase_bodies.items()
    )
    for p in patches:
        p.start()
    try:
        worker.run()
    finally:
        for p in patches:
            p.stop()


def test_worker_records_every_named_phase_in_order(worker) -> None:
    probe = ProviderProbeResult("tools", True, "", "ok", Path("x"), True)
    _run_worker(
        worker,
        _load_registry=lambda: "registry",
        _load_engine_manager=lambda: "engines",
        _probe_docker=lambda: False,
        _probe_tools_provider=lambda: probe,
    )
    names = [record.name for record in worker.results.phases]
    assert names == [PHASE_REGISTRY, PHASE_ENGINES, PHASE_DOCKER, PHASE_TOOLS_PROVIDER]
    assert all(r.outcome == OUTCOME_OK for r in worker.results.phases)
    assert worker.results.tools_provider is probe
    assert worker.results.docker_available is False
    assert worker.results.is_degraded is False
    assert worker.results.startup_time_ms >= 0
    worker.finished_signal.emit.assert_called_once_with(worker.results)
    messages = [call.args[0] for call in worker.progress_signal.emit.call_args_list]
    assert messages[0].startswith(PHASE_REGISTRY)
    assert messages[-1] == "Ready"


def test_worker_degrades_when_provider_probe_never_completes(worker) -> None:
    gate = threading.Event()
    try:
        with patch("src.launchers.startup.PROVIDER_PHASE_TIMEOUT_S", 0.05):
            _run_worker(
                worker,
                _load_registry=lambda: "registry",
                _load_engine_manager=lambda: None,
                _probe_docker=lambda: True,
                _probe_tools_provider=gate.wait,
            )
    finally:
        gate.set()
    worker.finished_signal.emit.assert_called_once()
    worker.error_signal.emit.assert_not_called()
    provider = worker.results.phases[-1]
    assert provider.name == PHASE_TOOLS_PROVIDER
    assert provider.outcome == OUTCOME_TIMEOUT
    assert provider.category == CATEGORY_TIMEOUT
    assert worker.results.tools_provider is None
    assert worker.results.is_degraded is True
    assert PHASE_TOOLS_PROVIDER in worker.results.degraded_summary()
    assert (
        "blocking phase: " + PHASE_TOOLS_PROVIDER in worker.results.format_diagnostics()
    )


def test_worker_degrades_when_provider_probe_raises(worker) -> None:
    def broken() -> None:
        raise ImportError("rate_of_closure is broken")

    _run_worker(
        worker,
        _load_registry=lambda: "registry",
        _load_engine_manager=lambda: None,
        _probe_docker=lambda: True,
        _probe_tools_provider=broken,
    )
    worker.finished_signal.emit.assert_called_once()
    assert worker.results.phases[-1].category == "import_failure"
    assert "rate_of_closure is broken" in worker.results.phases[-1].detail


def test_worker_reports_missing_provider_as_degraded_not_error(worker) -> None:
    missing = ProviderProbeResult("tools", False, CATEGORY_MISSING_CHECKOUT, "absent")
    _run_worker(
        worker,
        _load_registry=lambda: "registry",
        _load_engine_manager=lambda: None,
        _probe_docker=lambda: True,
        _probe_tools_provider=lambda: missing,
    )
    worker.finished_signal.emit.assert_called_once()
    worker.error_signal.emit.assert_not_called()
    assert worker.results.phases[-1].outcome == OUTCOME_DEGRADED
    assert worker.results.phases[-1].category == CATEGORY_MISSING_CHECKOUT


def test_worker_registry_timeout_is_a_hard_failure_with_phase_name(worker) -> None:
    gate = threading.Event()
    try:
        with patch("src.launchers.startup.REGISTRY_PHASE_TIMEOUT_S", 0.05):
            _run_worker(worker, _load_registry=gate.wait)
    finally:
        gate.set()
    worker.finished_signal.emit.assert_not_called()
    worker.error_signal.emit.assert_called_once()
    message = worker.error_signal.emit.call_args.args[0]
    assert PHASE_REGISTRY in message and "timeout" in message
    assert "still running" not in worker.diagnostics_text()
    assert PHASE_REGISTRY in worker.diagnostics_text()


def test_worker_probes_real_tools_provider_with_env(
    worker, tmp_path, monkeypatch
) -> None:
    """The worker hands ``TOOLS_REPO_PATH`` to the canonical resolver."""
    tools = tmp_path / "Tools"
    (tools / "src" / "rate_of_closure").mkdir(parents=True)
    (tools / "src" / "rate_of_closure" / "__init__.py").write_text("")
    (tools / "src" / "rate_of_closure" / "launch_pyqt6.py").write_text("")
    monkeypatch.setenv("TOOLS_REPO_PATH", str(tools))
    result = worker._probe_tools_provider()
    assert result.available is True
    assert result.entry_point == tools / "src" / "rate_of_closure" / "launch_pyqt6.py"


# ---------------------------------------------------------------------------
# StartupResults diagnostics helpers
# ---------------------------------------------------------------------------


def test_startup_results_from_dict_carries_phases_and_provider() -> None:
    record = StartupPhaseRecord("p", "t", 1, OUTCOME_DEGRADED, CATEGORY_TIMEOUT)
    results = StartupResults.from_dict({"phases": [record], "tools_provider": "probe"})
    assert results.phases == (record,)
    assert results.tools_provider == "probe"
    assert results.is_degraded is True
    assert results.degraded_summary() == "Startup degraded - p: degraded [timeout]"
    assert StartupResults().degraded_summary() == ""


# ---------------------------------------------------------------------------
# SplashScreen degraded state
# ---------------------------------------------------------------------------


def test_splash_show_degraded_updates_message(splash) -> None:
    splash.progress = 42
    splash.show_degraded("Tools provider missing")
    assert splash.loading_message == "Startup problem: Tools provider missing"
    assert splash.progress == 42
    with pytest.raises(ValueError):
        splash.show_degraded("")


# ---------------------------------------------------------------------------
# StartupFailureDialog
# ---------------------------------------------------------------------------


def test_failure_dialog_actions_are_deterministic(qapp) -> None:
    dialog = StartupFailureDialog("boom", "line 1\nline 2")
    assert dialog.action == StartupFailureAction.CLOSE
    dialog.retry_button.click()
    assert dialog.action == StartupFailureAction.RETRY
    dialog.continue_button.click()
    assert dialog.action == StartupFailureAction.CONTINUE
    dialog.close_button.click()
    assert dialog.action == StartupFailureAction.CLOSE
    dialog.reject()  # Escape path
    assert dialog.action == StartupFailureAction.CLOSE
    assert dialog.diagnostics_view.toPlainText() == "line 1\nline 2"
    dialog.deleteLater()


def test_failure_dialog_copy_diagnostics_uses_clipboard(qapp) -> None:
    dialog = StartupFailureDialog("boom", "copy me")
    dialog.copy_button.click()
    assert QApplication.clipboard().text() == "copy me"
    assert dialog.action == StartupFailureAction.CLOSE, "copy never closes"
    dialog.deleteLater()


def test_failure_dialog_is_keyboard_accessible(qapp) -> None:
    dialog = StartupFailureDialog("boom", "")
    for button in (
        dialog.retry_button,
        dialog.continue_button,
        dialog.copy_button,
        dialog.close_button,
    ):
        assert "&" in button.text(), f"{button.text()} lacks a mnemonic"
        assert button.focusPolicy().value & 0x1, "button must be Tab-focusable"
    assert dialog.continue_button.isDefault()
    assert dialog.diagnostics_view.accessibleName() == "Startup diagnostics"
    dialog.deleteLater()


def test_failure_dialog_validates_arguments(qapp) -> None:
    with pytest.raises(ValueError):
        StartupFailureDialog("", "")
    with pytest.raises(TypeError):
        StartupFailureDialog("boom", None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Launcher constructor keeps startup work off the GUI thread in loading mode
# ---------------------------------------------------------------------------


def test_loading_mode_does_not_load_registry_on_gui_thread(qapp) -> None:
    """``UpstreamDriftLauncher(loading=True)`` must leave loading to the worker.

    Before #8360 the constructor fell back to a synchronous registry and
    engine-manager load while the splash was visible, so a slow or hung
    import froze the splash with no timer able to fire.
    """
    from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher

    with (
        patch("src.launchers.upstream_drift_launcher.DockerCheckThread"),
        patch(
            "src.launchers.launcher_orchestrator._lazy_load_model_registry"
        ) as registry_loader,
        patch(
            "src.launchers.launcher_orchestrator._lazy_load_engine_manager"
        ) as engine_loader,
        patch("src.launchers.upstream_drift_launcher.QTimer"),
    ):
        launcher = UpstreamDriftLauncher(loading=True)
        try:
            registry_loader.assert_not_called()
            engine_loader.assert_not_called()
            assert launcher.loading is True
            assert launcher.registry is None
        finally:
            launcher.close()
            launcher.deleteLater()
            _spin(1)
