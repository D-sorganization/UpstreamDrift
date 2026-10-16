"""Launcher robustness: error hook, shutdown, launch resume, engine status.

Covers UD #12886, #12894, #12892, #12890, #12891 (epic #12878).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.launchers import engine_availability as ea
from src.launchers import upstream_drift_launcher_main as main_mod
from src.launchers.launcher_simulation import (
    DEPENDENCY_MAP,
    SimulationManager,
)
from src.launchers.model_card import DraggableModelCard

# ---- #12891 / #12890: engine availability probe ---------------------------


@pytest.mark.unit
def test_hint_table_covers_every_engine() -> None:
    assert set(ea.ENGINE_INSTALL_HINTS) == set(ea.ENGINE_MODULES)
    for engine in ea.ENGINE_MODULES:
        for plat in ("win32", "linux", "darwin"):
            assert ea.install_hint(engine, plat).strip()


@pytest.mark.unit
def test_pinocchio_hint_is_pin_not_pin_project() -> None:
    assert ea.install_hint("pinocchio", "linux") == "pip install pin"
    assert "pin-project" not in ea.install_hint("pinocchio", "win32")
    assert "pin-project" not in DEPENDENCY_MAP["pinocchio"]["install_cmd"]


@pytest.mark.unit
@pytest.mark.parametrize("engine", ["drake", "pinocchio"])
def test_windows_hints_point_to_wsl_docker(engine: str) -> None:
    hint = ea.install_hint(engine, "win32")
    assert "Docker" in hint and "WSL" in hint


@pytest.mark.unit
def test_opensim_hint_is_pip() -> None:
    assert ea.install_hint("opensim", "win32") == "pip install opensim"


@pytest.mark.unit
def test_probe_returns_frozen_result() -> None:
    res = ea.probe_engine("drake", platform="win32", module_present=lambda m: False)
    assert res.status == "unavailable" and not res.available
    assert "pydrake" in res.reason and "WSL" in res.install_hint
    with pytest.raises(AttributeError):
        res.available = True  # type: ignore[misc]
    ok = ea.probe_engine("drake", module_present=lambda m: True)
    assert ok.status == "available" and ok.reason == ""


@pytest.mark.unit
def test_probe_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        ea.probe_engine("nope")
    with pytest.raises(ValueError):
        ea.install_hint("nope")


@pytest.mark.unit
def test_engine_for_model_maps_ids_and_types() -> None:
    assert ea.engine_for_model("drake_golf", "drake") == "drake"
    assert ea.engine_for_model("x", "myosim") == "myosuite"
    assert ea.engine_for_model("mujoco_unified", "mujoco") is None


def _tile() -> Any:
    class _Stub:
        _STATUS_STRINGS = DraggableModelCard._STATUS_STRINGS

        def __init__(self) -> None:
            self.model = SimpleNamespace(
                id="drake_golf", type="drake", status="ready", launcher=None
            )

        def _target_is_resolvable(self) -> bool:
            return True

    return _Stub()


@pytest.mark.unit
def test_tile_badge_not_ready_when_engine_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.launchers.model_card.probe_engine",
        lambda engine: ea.probe_engine(engine, module_present=lambda m: False),
    )
    text, css = DraggableModelCard._get_status_info(_tile())
    assert text != "Ready" and css == "error"


@pytest.mark.unit
def test_tile_badge_ready_when_engine_present(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.launchers.model_card.probe_engine",
        lambda engine: ea.probe_engine(engine, module_present=lambda m: True),
    )
    text, _ = DraggableModelCard._get_status_info(_tile())
    assert text == "Ready"


# ---- #12886: exception hook -----------------------------------------------


def _exc_info(exc: BaseException) -> Any:
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001
        return (type(caught), caught, caught.__traceback__)


@pytest.mark.unit
def test_recoverable_error_keeps_app_alive_and_logs_to_log_dir(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    shown: list[tuple[str, str, str]] = []
    main_mod.mark_event_loop_running(True)
    try:
        log_dir = tmp_path / "logs"
        quit_app = main_mod.handle_uncaught_exception(
            *_exc_info(RuntimeError("boom")),
            show_dialog=lambda t, m, d: shown.append((t, m, d)),
            log_dir=log_dir,
        )
    finally:
        main_mod.mark_event_loop_running(False)
    assert quit_app is False
    assert shown and "Crash" not in shown[0][0] and "boom" in shown[0][2]
    assert (log_dir / "crash_traceback.txt").exists()
    assert not (tmp_path / "crash_traceback.txt").exists()


@pytest.mark.unit
def test_startup_error_is_fatal(tmp_path) -> None:
    main_mod.mark_event_loop_running(False)
    shown: list[Any] = []
    fatal = main_mod.handle_uncaught_exception(
        *_exc_info(ValueError("early")),
        show_dialog=lambda *a: shown.append(a),
        log_dir=tmp_path,
    )
    assert fatal is True
    assert shown[0][0] == "Application Crash"


@pytest.mark.unit
def test_dialog_failure_never_raises(tmp_path) -> None:
    main_mod.mark_event_loop_running(True)

    def bad(*_a: str) -> None:
        raise RuntimeError("dialog broke")

    try:
        result = main_mod.handle_uncaught_exception(
            *_exc_info(KeyError("k")), show_dialog=bad, log_dir=tmp_path
        )
    finally:
        main_mod.mark_event_loop_running(False)
    assert result is False


# ---- #12894: shutdown after worker deleted --------------------------------


@pytest.mark.unit
def test_shutdown_after_worker_delete_later(qapp) -> None:
    from PyQt6.QtCore import QCoreApplication, QEvent, QThread

    from src.launchers.startup_session import StartupSession

    class _Sig:
        def connect(self, *_a: Any) -> None:
            return

    class _W(QThread):
        def __init__(self) -> None:
            super().__init__()
            self.progress_signal = _Sig()
            self.finished_signal = _Sig()
            self.error_signal = _Sig()

        def run(self) -> None:
            return

    session = StartupSession(splash=None, shell=SimpleNamespace(), worker_factory=_W)
    session.start()
    worker = session.worker
    assert worker is not None
    from PyQt6 import sip

    worker.wait(2000)
    for _ in range(20):
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        if sip.isdeleted(worker):
            break
    assert sip.isdeleted(worker)
    session.shutdown()


# ---- #12892: probe completion resumes the pending launch ------------------


def _manager(pending: str | None, errors: list[Any]) -> tuple[Any, Any, list[str]]:
    model = SimpleNamespace(id="opensim_golf", name="OpenSim", type="opensim")
    launched: list[str] = []
    launcher = SimpleNamespace(
        selected_model="opensim_golf",
        _get_model=lambda mid: model,
        _dependency_status_cache={},
        _pending_launch_model_id=pending,
        _set_dependency_error_status=lambda *a: errors.append(a[2]),
    )
    mgr = SimulationManager(launcher)
    mgr.launch_simulation = lambda: launched.append("go")  # type: ignore[method-assign]
    return mgr, launcher, launched


@pytest.mark.unit
def test_probe_success_resumes_pending_launch() -> None:
    mgr, launcher, launched = _manager("opensim_golf", [])
    mgr._on_dependency_probe_finished("opensim_golf", True, "")
    assert launched == ["go"]
    assert launcher._pending_launch_model_id is None
    # No pending click (probe started by selection): must not launch.
    mgr._on_dependency_probe_finished("opensim_golf", True, "")
    assert launched == ["go"]


@pytest.mark.unit
def test_probe_failure_clears_pending_without_launch() -> None:
    errors: list[Any] = []
    mgr, launcher, launched = _manager("opensim_golf", errors)
    mgr._on_dependency_probe_finished("opensim_golf", False, "missing")
    assert launched == [] and errors == ["missing"]
    assert launcher._pending_launch_model_id is None
