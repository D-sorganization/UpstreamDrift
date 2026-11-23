"""Headless PyQt6 end-to-end tests for the Golf Suite Launcher (#15522).

Existing launcher tests call the private ``_launch_*`` methods directly. These
tests instead construct the real ``UpstreamDriftLauncher`` window and drive it
the way a user does: real ``QTest`` mouse clicks on the three launch buttons.
Only the process boundary (``subprocess.Popen``) and the modal error dialog are
replaced, so no model GUI, Docker container or blocking dialog is ever started.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton  # noqa: E402

from src.engines.physics_engines.mujoco.python import (  # noqa: E402
    golf_suite_launcher as gsl,
)

pytestmark = pytest.mark.unit

_APP: QApplication | None = None
_DRAKE_SCRIPT = "UpstreamDrift/engines/physics_engines/drake/python/src/golf_gui.py"


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv[:1])
    _APP = app  # keep a module-level reference alive
    return app


class _Recorder:
    """Captures ``Popen`` invocations and ``QMessageBox.critical`` dialogs."""

    def __init__(self) -> None:
        self.popen_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.dialogs: list[tuple[str, str]] = []
        self.popen_error: Exception | None = None


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Real shown launcher whose script paths point at throwaway files."""
    _ensure_qapp()
    rec = _Recorder()

    def fake_popen(cmd: list[str], **kwargs: Any) -> object:
        if rec.popen_error is not None:
            raise rec.popen_error
        rec.popen_calls.append((cmd, kwargs))
        return object()

    def fake_critical(_parent: Any, title: str, text: str, *_a: Any) -> int:
        rec.dialogs.append((title, text))
        return int(QMessageBox.StandardButton.Ok.value)

    monkeypatch.setattr(gsl.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gsl.QtWidgets.QMessageBox, "critical", fake_critical)

    window = gsl.UpstreamDriftLauncher()
    for attr in ("mujoco_path", "pinocchio_path"):
        script = tmp_path / attr / "run.py"
        script.parent.mkdir(parents=True)
        script.write_text("pass\n", encoding="utf-8")
        setattr(window, attr, script)
    window.show()
    window.rec = rec  # type: ignore[attr-defined]
    yield window
    window.close()
    window.deleteLater()


def _click(button: QPushButton) -> None:
    assert button.isEnabled() and button.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def test_window_launches_with_three_enabled_buttons_and_ready_status(launcher) -> None:
    """A fresh launcher shows its title, three launch buttons and "Ready"."""
    assert launcher.windowTitle() == "Golf Model Suite Launcher"
    assert launcher.status.text() == "Ready"
    buttons = launcher.findChildren(QPushButton)
    assert [b.text() for b in buttons] == [
        "Launch MuJoCo Model",
        "Launch Drake Model",
        "Launch Pinocchio Model",
    ]
    assert all(b.isEnabled() and b.minimumHeight() == 40 for b in buttons)
    assert launcher.rec.popen_calls == []


@pytest.mark.parametrize(
    ("button_attr", "path_attr", "name"),
    [
        ("btn_mujoco", "mujoco_path", "MuJoCo"),
        ("btn_pinocchio", "pinocchio_path", "Pinocchio"),
    ],
)
def test_clicking_script_button_spawns_interpreter_and_updates_status(
    launcher, button_attr: str, path_attr: str, name: str
) -> None:
    """A click starts the model script with this interpreter in its work dir."""
    script: Path = getattr(launcher, path_attr)
    _click(getattr(launcher, button_attr))
    assert len(launcher.rec.popen_calls) == 1
    cmd, kwargs = launcher.rec.popen_calls[0]
    assert cmd == [sys.executable, str(script)]
    assert kwargs["cwd"] == str(script.parent.parent)
    assert launcher.status.text() == f"{name} Launched"
    assert launcher.rec.dialogs == []


@pytest.mark.parametrize(
    ("button_attr", "path_attr"),
    [("btn_mujoco", "mujoco_path"), ("btn_pinocchio", "pinocchio_path")],
)
def test_missing_script_shows_error_dialog_and_never_spawns(
    launcher, button_attr: str, path_attr: str
) -> None:
    """A vanished script reports the path in a dialog and sets "Error"."""
    missing = launcher.mujoco_path.parent / "gone.py"
    setattr(launcher, path_attr, missing)
    _click(getattr(launcher, button_attr))
    assert launcher.rec.popen_calls == []
    assert launcher.status.text() == "Error"
    assert len(launcher.rec.dialogs) == 1
    title, text = launcher.rec.dialogs[0]
    assert title == "Error"
    assert str(missing) in text


def test_spawn_failure_is_reported_not_raised(launcher) -> None:
    """An OSError from the OS surfaces as a dialog and an Error status."""
    launcher.rec.popen_error = OSError("exec format error")
    _click(launcher.btn_mujoco)
    assert launcher.status.text() == "Error"
    assert len(launcher.rec.dialogs) == 1
    text = launcher.rec.dialogs[0][1]
    assert "Failed to launch MuJoCo" in text
    assert "exec format error" in text
    # The window stays usable: a later successful click recovers the status.
    launcher.rec.popen_error = None
    _click(launcher.btn_pinocchio)
    assert launcher.status.text() == "Pinocchio Launched"


def test_drake_click_builds_docker_command_mounting_repositories(launcher) -> None:
    """The Drake button runs the engine image with the repos dir mounted."""
    _click(launcher.btn_drake)
    assert len(launcher.rec.popen_calls) == 1
    cmd, kwargs = launcher.rec.popen_calls[0]
    assert cmd[:4] == ["docker", "run", "--rm", "-it"]
    assert "upstream-drift:engine" in cmd
    assert cmd[cmd.index("-v") + 1] == f"{launcher.repos_dir.resolve()}:/workspace"
    if sys.platform.startswith("linux"):
        assert "--net=host" in cmd
    else:
        assert "--net=host" not in cmd
        assert "DISPLAY=host.docker.internal:0" in cmd
    assert cmd[-2:] == ["python", _DRAKE_SCRIPT]
    assert kwargs["cwd"] == str(launcher.repo_root)
    assert launcher.status.text() == "Drake Launched (Docker)"


def test_drake_missing_docker_reports_error_dialog(launcher) -> None:
    """Without Docker installed the click ends in a dialog, not a crash."""
    launcher.rec.popen_error = FileNotFoundError("docker")
    _click(launcher.btn_drake)
    assert launcher.status.text() == "Error"
    assert len(launcher.rec.dialogs) == 1
    assert "Failed to launch Drake Docker container" in launcher.rec.dialogs[0][1]


def test_subprocess_error_type_is_also_handled(launcher) -> None:
    """``SubprocessError`` (not only ``OSError``) is caught at the boundary."""
    launcher.rec.popen_error = subprocess.SubprocessError("boom")
    _click(launcher.btn_pinocchio)
    assert launcher.status.text() == "Error"
    assert len(launcher.rec.dialogs) == 1
