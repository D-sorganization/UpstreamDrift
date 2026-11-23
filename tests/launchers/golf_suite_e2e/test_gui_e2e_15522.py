"""Real QTest launcher flows; intercept only process and modal-dialog boundaries.

No engine process, Docker daemon, X server or native rendering is started.
These tests qualify launcher dispatch and recovery, not engine dynamics.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from src.engines.physics_engines.mujoco.python import golf_suite_launcher as gui

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def _click(button: QWidget) -> None:
    """Call the static Qt namespace function despite old wheel stub annotations."""
    click = cast(Callable[[QWidget, Qt.MouseButton], None], QTest.mouseClick)
    click(button, Qt.MouseButton.LeftButton)


@pytest.fixture(params=[(1280, 900), (390, 844)], ids=["desktop", "phone"])
def launcher(
    qapp: QApplication,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[tuple[gui.UpstreamDriftLauncher, MagicMock, MagicMock]]:
    """Show the real window and always dispose it, including on assertion failure."""
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(gui.subprocess, "Popen") as launch,
        patch.object(gui.QtWidgets.QMessageBox, "critical") as critical,
    ):
        window = gui.UpstreamDriftLauncher()
        window.resize(*request.param)
        window.show()
        qapp.processEvents()
        assert window.isVisible()
        assert window.status.text() == "Ready"
        try:
            yield window, launch, critical
        finally:
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            assert sip.isdeleted(window)


@pytest.mark.parametrize("engine", ["mujoco", "pinocchio"])
def test_local_model_launch_and_relaunch(
    launcher: tuple[gui.UpstreamDriftLauncher, MagicMock, MagicMock], engine: str
) -> None:
    """Clicks route to existing in-tree entry points with package imports intact."""
    window, launch, critical = launcher
    package = ROOT / "src/engines/physics_engines" / engine / "python"
    script = package / (
        "mujoco_humanoid_golf/__main__.py"
        if engine == "mujoco"
        else "pinocchio_golf/gui.py"
    )
    assert script.is_file()
    button = getattr(window, f"btn_{engine}")
    for count in (1, 2):
        _click(button)
        assert launch.call_count == count
        module = f"src.engines.physics_engines.{engine}.python." + (
            "mujoco_humanoid_golf" if engine == "mujoco" else "pinocchio_golf.gui"
        )
        launch.assert_called_with([sys.executable, "-m", module], cwd=str(ROOT))
        assert window.status.text() == f"{engine.title()} Launched".replace(
            "Mujoco", "MuJoCo"
        )
        critical.assert_not_called()


def test_drake_container_launch(
    launcher: tuple[gui.UpstreamDriftLauncher, MagicMock, MagicMock],
) -> None:
    """A real button click builds the consolidated-repository Docker command."""
    window, launch, critical = launcher
    _click(window.btn_drake)
    launch.assert_called_once()
    args, kwargs = launch.call_args
    command = args[0]
    assert command[:4] == ["docker", "run", "--rm", "-it"]
    assert command[-3:] == [
        "python",
        "-m",
        "src.engines.physics_engines.drake.python.src.drake_gui_app",
    ]
    assert command[command.index("-v") + 1] == f"{ROOT.parent}:/workspace"
    assert command[command.index("-w") + 1] == f"/workspace/{ROOT.name}"
    assert kwargs == {"cwd": str(ROOT)}
    assert window.status.text() == "Drake Launched (Docker)"
    critical.assert_not_called()


def test_drake_target_is_runnable(
    launcher: tuple[gui.UpstreamDriftLauncher, MagicMock, MagicMock],
) -> None:
    """Never route Drake to the historical docstring-only golf_gui placeholder."""
    window, _, _ = launcher
    tree = ast.parse(window.drake_path.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
    )


def test_missing_script_then_recovery(
    launcher: tuple[gui.UpstreamDriftLauncher, MagicMock, MagicMock], tmp_path: Path
) -> None:
    """Missing scripts report failure without spawning; the next click can recover."""
    window, launch, critical = launcher
    original = window.mujoco_path
    window.mujoco_path = tmp_path / "missing.py"
    _click(window.btn_mujoco)
    assert window.status.text() == "Error"
    launch.assert_not_called()
    critical.assert_called_once()
    assert "Could not find script" in critical.call_args.args[2]
    window.mujoco_path = original
    _click(window.btn_mujoco)
    launch.assert_called_once()
    assert window.status.text() == "MuJoCo Launched"


@pytest.mark.parametrize("engine", ["mujoco", "drake"])
@pytest.mark.parametrize(
    "error", [OSError("unavailable"), subprocess.SubprocessError("failed")]
)
def test_process_failure_then_recovery(
    launcher: tuple[gui.UpstreamDriftLauncher, MagicMock, MagicMock],
    engine: str,
    error: Exception,
) -> None:
    """Launch failures cannot leave a success badge or break later button use."""
    window, launch, critical = launcher
    button = getattr(window, f"btn_{engine}")
    launch.side_effect = error
    _click(button)
    assert window.status.text() == "Error"
    critical.assert_called_once()
    assert str(error) in critical.call_args.args[2]
    launch.side_effect = None
    _click(button)
    assert launch.call_count == 2
    assert "Launched" in window.status.text()
