"""Tests for golf_suite_launcher.py."""

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def mock_pyqt(qapp):
    """Mock PyQt6 components for testing non-UI logic safely."""
    with (
        patch("src.launchers.golf_suite_launcher.PYQT6_AVAILABLE", True),
        patch("src.launchers.golf_suite_launcher.QtWidgets") as mock_widgets,
        patch("src.launchers.golf_suite_launcher.QtCore") as mock_core,
        patch("src.launchers.golf_suite_launcher.QtGui"),
    ):
        mock_widgets.QMainWindow = MagicMock
        mock_widgets.QApplication.clipboard.return_value = MagicMock()
        yield mock_widgets, mock_core


@pytest.fixture
def launcher(mock_pyqt):
    """Provide a minimal instantiated GolfLauncher."""
    with patch("src.launchers.golf_suite_launcher.GolfLauncher._setup_ui"):
        from src.launchers.golf_suite_launcher import GolfLauncher

        inst = GolfLauncher()
        inst.status = MagicMock()
        inst.log_text = MagicMock()
        inst.clear_btn = MagicMock()
        inst.copy_btn = MagicMock()
        inst.style = MagicMock()
        yield inst


def test_init_raises_without_pyqt():
    with patch("src.launchers.golf_suite_launcher.PYQT6_AVAILABLE", False):
        import src.launchers.golf_suite_launcher as gsl

        with pytest.raises(ImportError, match="PyQt6 is required"):
            gsl.GolfLauncher()


def test_imports_without_pyqt():
    import importlib

    import src.launchers.golf_suite_launcher as gsl

    with patch(
        "src.shared.python.engine_core.engine_availability.PYQT6_AVAILABLE", False
    ):
        try:
            importlib.reload(gsl)
            assert gsl.QtWidgets is None
            assert gsl.QtCore is None
            assert gsl.QtGui is None
        finally:
            # Restore module state so other tests on this worker do not fail!
            pass

    # Always reload after the patch has ended to restore true state
    importlib.reload(gsl)


def test_init_sets_paths(mock_pyqt):
    from src.launchers.golf_suite_launcher import GolfLauncher

    with patch.object(GolfLauncher, "_setup_ui"):
        launcher = GolfLauncher()
        assert hasattr(launcher, "suite_root")
        assert hasattr(launcher, "mujoco_path")
        assert hasattr(launcher, "drake_path")
        assert "engines" in str(launcher.mujoco_path)


def test_setup_ui_execution(qapp):
    from src.launchers.golf_suite_launcher import GolfLauncher

    # Do not patch _setup_ui, let it execute with real PyQT
    launcher = GolfLauncher()

    # Verify that UI components were created
    assert hasattr(launcher, "status")
    assert hasattr(launcher, "log_text")
    assert hasattr(launcher, "btn_mujoco")
    assert hasattr(launcher, "btn_drake")
    assert hasattr(launcher, "btn_pinocchio")
    assert hasattr(launcher, "btn_opensim")
    assert hasattr(launcher, "btn_myosim")
    assert hasattr(launcher, "btn_openpose")
    assert hasattr(launcher, "btn_urdf")
    assert hasattr(launcher, "btn_shot_tracer")
    assert hasattr(launcher, "copy_btn")
    assert hasattr(launcher, "clear_btn")


def test_launch_script_success(launcher):
    fake_path = Path("fake/path.py")
    fake_cwd = Path("fake/cwd")

    with (
        patch("src.launchers.golf_suite_launcher.Path.exists", return_value=True),
        patch("src.launchers.golf_suite_launcher.subprocess.Popen") as mock_popen,
    ):
        mock_process = MagicMock()
        mock_process.pid = 1234
        mock_popen.return_value = mock_process

        launcher._launch_script("Test Engine", fake_path, fake_cwd)

        mock_popen.assert_called_once_with(
            [sys.executable, str(fake_path)], cwd=str(fake_cwd)
        )
        launcher.status.setText.assert_called_with("Test Engine Launched")


def test_launch_script_not_found(launcher):
    fake_path = Path("fake/path.py")
    fake_cwd = Path("fake/cwd")

    with (
        patch("src.launchers.golf_suite_launcher.Path.exists", return_value=False),
        patch("src.launchers.golf_suite_launcher.QtWidgets.QMessageBox") as mock_msgbox,
    ):
        launcher._launch_script("Test Engine", fake_path, fake_cwd)
        mock_msgbox.critical.assert_called_once()
        launcher.status.setText.assert_called_with("Error: Script not found")


def test_launch_script_subprocess_error(launcher):
    fake_path = Path("fake/path.py")
    fake_cwd = Path("fake/cwd")

    with (
        patch("src.launchers.golf_suite_launcher.Path.exists", return_value=True),
        patch(
            "src.launchers.golf_suite_launcher.subprocess.Popen",
            side_effect=OSError("Boom"),
        ),
        patch("src.launchers.golf_suite_launcher.QtWidgets.QMessageBox") as mock_msgbox,
    ):
        launcher._launch_script("Test Engine", fake_path, fake_cwd)
        mock_msgbox.critical.assert_called_once()
        launcher.status.setText.assert_called_with("Error")


def test_log_message(launcher):
    launcher.log_message("Test message")
    launcher.log_text.append.assert_called_once()
    assert "Test message" in launcher.log_text.append.call_args[0][0]


def test_clear_log(launcher):
    launcher.clear_log()
    launcher.log_text.clear.assert_called_once()
    launcher.clear_btn.setText.assert_called_with("Cleared!")


def test_copy_log(launcher, mock_pyqt):
    mock_widgets, mock_core = mock_pyqt
    mock_clipboard = MagicMock()
    mock_widgets.QApplication.clipboard.return_value = mock_clipboard

    launcher.log_text.toPlainText.return_value = "Log content"
    launcher.copy_log()
    mock_clipboard.setText.assert_called_once_with("Log content")
    launcher.copy_btn.setText.assert_called_with("Copied!")


def test_copy_log_no_clipboard(launcher, mock_pyqt):
    mock_widgets, mock_core = mock_pyqt
    mock_widgets.QApplication.clipboard.return_value = None

    launcher.log_text.toPlainText.return_value = "Log content"
    launcher.copy_log()
    launcher.copy_btn.setText.assert_not_called()


def test_restore_btn(launcher):
    mock_btn = MagicMock()
    mock_icon = MagicMock()
    launcher._restore_btn(mock_btn, "Restored", mock_icon)
    mock_btn.setText.assert_called_once_with("Restored")
    mock_btn.setIcon.assert_called_once_with(mock_icon)


def test_restore_btn_none(launcher):
    # Tests the fallback conditions where btn=None or icon=None
    launcher._restore_btn(None, "Restored", MagicMock())

    mock_btn = MagicMock()
    launcher._restore_btn(mock_btn, "Restored", None)
    mock_btn.setText.assert_called_once_with("Restored")
    mock_btn.setIcon.assert_not_called()


def test_launcher_methods(launcher):
    with patch.object(launcher, "_launch_script") as mock_launch:
        launcher._launch_mujoco()
        assert "MuJoCo" in mock_launch.call_args[0]
        mock_launch.reset_mock()

        launcher._launch_drake()
        assert "Drake" in mock_launch.call_args[0]
        mock_launch.reset_mock()

        launcher._launch_pinocchio()
        assert "Pinocchio" in mock_launch.call_args[0]
        mock_launch.reset_mock()

        launcher._launch_opensim()
        assert "OpenSim" in mock_launch.call_args[0]
        mock_launch.reset_mock()

        launcher._launch_myosim()
        assert "MyoSim" in mock_launch.call_args[0]
        mock_launch.reset_mock()

        launcher._launch_openpose()
        assert "OpenPose" in mock_launch.call_args[0]
        mock_launch.reset_mock()

        launcher._launch_urdf()
        assert "URDF Generator" in mock_launch.call_args[0]
        mock_launch.reset_mock()

        launcher._launch_shot_tracer()
        assert "Shot Tracer" in mock_launch.call_args[0]


def test_main_no_pyqt():
    with patch("src.launchers.golf_suite_launcher.PYQT6_AVAILABLE", False):
        from src.launchers.golf_suite_launcher import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1


def test_main_with_pyqt(mock_pyqt):
    with (
        patch("src.launchers.golf_suite_launcher.PYQT6_AVAILABLE", True),
        patch("src.launchers.golf_suite_launcher.GolfLauncher"),
    ):
        from src.launchers.golf_suite_launcher import main

        mock_widgets, mock_core = mock_pyqt
        mock_app = MagicMock()
        mock_app.exec.return_value = 0
        mock_widgets.QApplication.return_value = mock_app
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
