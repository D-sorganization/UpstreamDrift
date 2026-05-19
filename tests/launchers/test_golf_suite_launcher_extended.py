"""Extended coverage for ``src.launchers.golf_suite_launcher``.

The pre-existing ``tests/launchers/test_golf_suite_launcher.py`` mocks out
all of ``QtWidgets`` so production code never actually runs.  This file
instantiates the real ``UpstreamDriftLauncher`` against the offscreen Qt platform
to exercise the UI scaffold (``_setup_ui``, ``_setup_engine_buttons``,
``_setup_shot_tracer_section``, ``_setup_log_area``), the log
helpers, and the ``_launch_script`` happy / sad paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.launchers import golf_suite_launcher as gsl


@pytest.fixture
def real_launcher(qapp):
    if not gsl.PYQT6_AVAILABLE:
        pytest.skip("PyQt6 unavailable")
    win = gsl.UpstreamDriftLauncher()
    yield win
    win.deleteLater()


def test_real_launcher_constructs(real_launcher) -> None:
    assert real_launcher.windowTitle() == ("Golf Modeling Suite - Local Launcher")
    # All engine launch buttons are present.
    for attr in (
        "btn_mujoco",
        "btn_drake",
        "btn_pinocchio",
        "btn_opensim",
        "btn_myosim",
        "btn_openpose",
        "btn_urdf",
        "btn_pendulum",
        "btn_shot_tracer",
        "log_text",
        "copy_btn",
        "clear_btn",
        "status",
    ):
        assert hasattr(real_launcher, attr)


def test_real_launcher_log_message_appends(real_launcher) -> None:
    real_launcher.log_text.clear()
    real_launcher.log_message("hello world")
    assert "hello world" in real_launcher.log_text.toPlainText()


def test_real_launcher_clear_log(real_launcher) -> None:
    real_launcher.log_message("noise")
    real_launcher.clear_log()
    # The implementation appends a "Log cleared" notice after clearing.
    text = real_launcher.log_text.toPlainText()
    assert "noise" not in text


def test_real_launcher_copy_log_uses_clipboard(real_launcher) -> None:
    real_launcher.log_message("payload")
    fake_clipboard = MagicMock()
    with patch.object(
        gsl.QtWidgets.QApplication,
        "clipboard",
        return_value=fake_clipboard,
    ):
        real_launcher.copy_log()
    fake_clipboard.setText.assert_called_once()
    assert "payload" in fake_clipboard.setText.call_args.args[0]


def test_real_launcher_launch_script_with_missing_path_shows_error(
    real_launcher, tmp_path
) -> None:
    missing = tmp_path / "nonexistent_engine.py"
    with patch.object(gsl.QtWidgets, "QMessageBox") as mb:
        real_launcher._launch_script("Test", missing, tmp_path)
    mb.critical.assert_called_once()


def test_real_launcher_launch_script_invokes_popen(real_launcher, tmp_path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    with patch.object(gsl.subprocess, "Popen") as mock_popen:
        mock_popen.return_value = MagicMock(pid=4242)
        real_launcher._launch_script("Test", script, tmp_path)

    mock_popen.assert_called_once()
    args = mock_popen.call_args.args[0]
    assert args[0] == sys.executable
    assert args[1] == str(script)


def test_real_launcher_launch_script_handles_subprocess_error(
    real_launcher, tmp_path
) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    with patch.object(
        gsl.subprocess,
        "Popen",
        side_effect=OSError("boom"),
    ):
        with patch.object(gsl.QtWidgets, "QMessageBox") as mb:
            real_launcher._launch_script("Test", script, tmp_path)
        mb.critical.assert_called_once()


def test_real_launcher_engine_slots_invoke_launch_script(real_launcher) -> None:
    """The engine-specific slots should each delegate to ``_launch_script``."""
    with patch.object(real_launcher, "_launch_script") as mock_launch:
        for slot_name in (
            "_launch_mujoco",
            "_launch_drake",
            "_launch_pinocchio",
            "_launch_opensim",
            "_launch_myosim",
            "_launch_openpose",
            "_launch_urdf",
            "_launch_pendulums",
            "_launch_shot_tracer",
        ):
            slot = getattr(real_launcher, slot_name)
            slot()
        assert mock_launch.call_count == 9


def test_module_pyqt6_available_flag_is_set() -> None:
    # Sanity check that running on a system with PyQt6 reports the flag.
    import importlib

    spec = importlib.util.find_spec("PyQt6")
    if spec is not None:
        assert gsl.PYQT6_AVAILABLE is True
