"""Tests for golf_suite_launcher.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Requires pytest-qt or an active QApplication instance.
# We will create a local QApplication if one does not exist.


@pytest.fixture(autouse=True)
def qapp():
    """Ensure a QApplication exists before tests run."""
    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        yield app
    except ImportError:
        yield None


@pytest.fixture
def launcher(qapp):
    """Provide a minimal instantiated GolfLauncher."""
    if not qapp:
        pytest.skip("PyQt6 is not available")
    from src.launchers.golf_suite_launcher import GolfLauncher

    inst = GolfLauncher()
    yield inst


def test_init_sets_paths(launcher):
    assert hasattr(launcher, "suite_root")
    assert hasattr(launcher, "mujoco_path")
    assert hasattr(launcher, "drake_path")
    assert "engines" in str(launcher.mujoco_path)


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

        # Test launching
        launcher._launch_script("Test Engine", fake_path, fake_cwd)

        mock_popen.assert_called_once_with(
            [sys.executable, str(fake_path)], cwd=str(fake_cwd)
        )
        assert "Test Engine Launched" in launcher.status.text()


def test_launch_script_not_found(launcher):
    fake_path = Path("fake/path.py")
    fake_cwd = Path("fake/cwd")

    with (
        patch("src.launchers.golf_suite_launcher.Path.exists", return_value=False),
        patch("src.launchers.golf_suite_launcher.QtWidgets.QMessageBox") as mock_msgbox,
    ):
        launcher._launch_script("Test Engine", fake_path, fake_cwd)
        mock_msgbox.critical.assert_called_once()
        assert "Error: Script not found" in launcher.status.text()


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
        assert "Error" in launcher.status.text()


def test_log_message(launcher):
    launcher.log_message("Test message")
    assert "Test message" in launcher.log_text.toPlainText()


def test_clear_log(launcher):
    launcher.log_message("To be cleared")
    launcher.clear_log()
    assert "Log cleared" in launcher.log_text.toPlainText()
    assert "Cleared!" in launcher.clear_btn.text()


def test_copy_log(launcher):
    with patch(
        "src.launchers.golf_suite_launcher.QtWidgets.QApplication.clipboard"
    ) as mock_clip:
        mock_clipboard = MagicMock()
        mock_clip.return_value = mock_clipboard

        launcher.log_message("Log content")
        launcher.copy_log()

        mock_clipboard.setText.assert_called_once()


def test_restore_btn(launcher):
    from PyQt6.QtWidgets import QPushButton

    btn = QPushButton("btn")

    launcher._restore_btn(btn, "Restored", None)
    assert btn.text() == "Restored"


def test_launcher_methods(launcher):
    """Test that all specific engine launchers call _launch_script correctly."""
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


def test_init_raises_without_pyqt():
    # We must reload the module to truly test this, or just mock PYQT6_AVAILABLE
    # and call __init__ directly on an object that overrides the super calls.
    # It checks PYQT6_AVAILABLE at module level and inside __init__.
    with patch("src.launchers.golf_suite_launcher.PYQT6_AVAILABLE", False):
        import src.launchers.golf_suite_launcher as gsl

        # We don't want to actually reload because it destroys classes.
        # Just mock GolfLauncher base class?
        # A simpler way: just test that if we call __init__ without it, it raises.
        # Unfortunately GolfLauncher is a QMainWindow, so calling __init__ is safe if
        # we bypass super().__init__() or catch the error early.
        with pytest.raises(ImportError, match="PyQt6 is required"):
            gsl.GolfLauncher.__init__(MagicMock())


def test_main_no_pyqt():
    with patch("src.launchers.golf_suite_launcher.PYQT6_AVAILABLE", False):
        from src.launchers.golf_suite_launcher import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1


def test_main_with_pyqt(qapp):
    if not qapp:
        pytest.skip("PyQt6 is not available")
    with (
        patch("src.launchers.golf_suite_launcher.PYQT6_AVAILABLE", True),
        patch("src.launchers.golf_suite_launcher.GolfLauncher.show"),
        patch(
            "src.launchers.golf_suite_launcher.QtWidgets.QApplication.exec",
            return_value=0,
        ),
    ):
        from src.launchers.golf_suite_launcher import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
