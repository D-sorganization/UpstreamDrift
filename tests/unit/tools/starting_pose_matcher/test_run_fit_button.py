"""Tests for the Run-fit QThread widget (issue #4707, slice 2/3)."""

from __future__ import annotations

import os

# MUST set the platform BEFORE any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip(
    "PyQt6",
    reason="PyQt6 required for Run-fit widget tests",
    exc_type=ImportError,
)
pytest.importorskip(
    "PyQt6.QtWidgets",
    reason="PyQt6.QtWidgets not loadable in this environment",
    exc_type=ImportError,
)

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.motion_matching import provider_registry  # noqa: E402
from src.tools.starting_pose_matcher.widgets.run_fit_button import (  # noqa: E402
    FitWorker,
    RunFitButton,
)


pytestmark = pytest.mark.unit


class _OkProvider:
    engine_name = "ok-engine"

    def __init__(self) -> None:
        self.calls: list[object] = []

    def fit_swing(self, target):
        self.calls.append(target)
        return {"target": target, "ok": True}


class _BoomProvider:
    engine_name = "boom-engine"

    def fit_swing(self, target):  # noqa: ARG002
        raise RuntimeError("kaboom")


class _SlowProvider:
    """Provider that polls the worker's cancel flag in a tight loop."""

    engine_name = "slow-engine"

    def __init__(self) -> None:
        self.worker_ref: FitWorker | None = None

    def fit_swing(self, target):  # noqa: ARG002
        import time

        for _ in range(20):
            if self.worker_ref is not None and self.worker_ref._cancelled:
                return {"interrupted": True}
            time.sleep(0.005)
        return {"slow": True}


@pytest.fixture
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def _clean_registry():
    provider_registry.clear_registry()
    yield provider_registry
    provider_registry.clear_registry()


def _spin_until(predicate, timeout_ms: int = 3000) -> bool:
    """Pump the Qt event loop until ``predicate()`` is true or we time out."""
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(20)
    deadline = {"left": timeout_ms}

    def _tick() -> None:
        if predicate() or deadline["left"] <= 0:
            timer.stop()
            loop.quit()
        deadline["left"] -= timer.interval()

    timer.timeout.connect(_tick)
    timer.start()
    loop.exec()
    return predicate()


def test_button_disabled_until_target_and_engine(_qapp, _clean_registry):
    w = RunFitButton()
    assert w.btn_run.isEnabled() is False
    w.set_inputs(target=None, engine_name="ok-engine")
    assert w.btn_run.isEnabled() is False
    w.set_inputs(target=object(), engine_name="")
    assert w.btn_run.isEnabled() is False
    w.set_inputs(target=object(), engine_name="ok-engine")
    assert w.btn_run.isEnabled() is True


def test_start_fit_validates_inputs(_qapp, _clean_registry):
    w = RunFitButton()
    with pytest.raises(ValueError, match="no target"):
        w.start_fit()
    w.set_inputs(target=object(), engine_name="")
    with pytest.raises(ValueError, match="no engine"):
        w.start_fit()


def test_fit_worker_rejects_bad_args():
    with pytest.raises(ValueError):
        FitWorker("", object())
    with pytest.raises(ValueError):
        FitWorker("ok-engine", None)


def test_success_path_emits_result(_qapp, _clean_registry):
    provider = _OkProvider()
    _clean_registry.register_provider(provider)
    w = RunFitButton()
    target = {"sentinel": True}
    w.set_inputs(target=target, engine_name="ok-engine")
    received: list[object] = []
    w.finished.connect(received.append)
    w.start_fit()
    assert _spin_until(lambda: bool(received))
    assert received and received[0] == {"target": target, "ok": True}
    assert provider.calls == [target]
    assert w.last_result == {"target": target, "ok": True}
    assert w.btn_cancel.isEnabled() is False
    assert w.btn_run.isEnabled() is True
    assert "complete" in w.lbl_status.text().lower()


def test_failure_path_emits_error(_qapp, _clean_registry):
    _clean_registry.register_provider(_BoomProvider())
    w = RunFitButton()
    w.set_inputs(target=object(), engine_name="boom-engine")
    errors: list[str] = []
    w.failed.connect(errors.append)
    w.start_fit()
    assert _spin_until(lambda: bool(errors))
    assert errors and "kaboom" in errors[0]
    assert w.last_result is None
    assert "fail" in w.lbl_status.text().lower()
    assert w.btn_run.isEnabled() is True


def test_unknown_engine_emits_error(_qapp, _clean_registry):
    w = RunFitButton()
    w.set_inputs(target=object(), engine_name="missing-engine")
    errors: list[str] = []
    w.failed.connect(errors.append)
    w.start_fit()
    assert _spin_until(lambda: bool(errors))
    assert errors and "missing-engine" in errors[0]


def test_cancel_terminates_thread_cleanly(_qapp, _clean_registry):
    provider = _SlowProvider()
    _clean_registry.register_provider(provider)
    w = RunFitButton()
    w.set_inputs(target=object(), engine_name="slow-engine")
    w.start_fit()
    provider.worker_ref = w._worker
    assert w.btn_cancel.isEnabled() is True
    w.cancel()
    assert _spin_until(lambda: w._thread is None)
    assert w._thread is None
    assert w._worker is None
    assert w.btn_cancel.isEnabled() is False
    assert "cancel" in w.lbl_status.text().lower()


def test_cannot_start_while_running(_qapp, _clean_registry):
    _clean_registry.register_provider(_SlowProvider())
    w = RunFitButton()
    w.set_inputs(target=object(), engine_name="slow-engine")
    w.start_fit()
    try:
        with pytest.raises(ValueError, match="already running"):
            w.start_fit()
    finally:
        w.cancel()
        _spin_until(lambda: w._thread is None)
