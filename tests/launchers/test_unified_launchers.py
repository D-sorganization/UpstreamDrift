"""Tests for unified and mocap launchers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.launchers.motion_capture_launcher import MoCapLauncher
from src.launchers.motion_capture_launcher import main as mocap_main
from src.launchers.mujoco_unified_launcher import MujocoUnifiedLauncher
from src.launchers.mujoco_unified_launcher import main as mujoco_unified_main


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
def test_mocap_launcher_items(mock_base_init):
    launcher = MoCapLauncher()
    items = launcher.get_items()
    assert len(items) == 3
    for item in items:
        assert item.name
        assert item.item_type == "tool"


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_mocap_launcher_python_launch_success(mock_exists, mock_popen, mock_base_init):
    launcher = MoCapLauncher()
    launcher.show_error = MagicMock()

    launcher._launch_python_script("fake/path.py")
    mock_popen.assert_called_once()
    launcher.show_error.assert_not_called()


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
@patch.object(Path, "exists", return_value=False)
def test_mocap_launcher_python_launch_not_found(mock_exists, mock_base_init):
    launcher = MoCapLauncher()
    launcher.show_error = MagicMock()

    launcher._launch_python_script("fake/path.py")
    launcher.show_error.assert_called_once()


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
@patch("subprocess.Popen", side_effect=OSError("Failed"))
@patch.object(Path, "exists", return_value=True)
def test_mocap_launcher_python_launch_os_error(mock_exists, mock_popen, mock_base_init):
    launcher = MoCapLauncher()
    launcher.show_error = MagicMock()

    launcher._launch_python_script("fake/path.py")
    launcher.show_error.assert_called_once()


def test_mocap_main():
    with patch("src.launchers.motion_capture_launcher.run_launcher") as mock_run:
        mock_run.return_value = 0
        assert mocap_main() == 0
        mock_run.assert_called_once_with(MoCapLauncher)


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
def test_mujoco_unified_launcher_items(mock_base_init):
    launcher = MujocoUnifiedLauncher()
    items = launcher.get_items()
    assert len(items) == 2
    for item in items:
        assert item.name
        assert item.item_type == "tool"


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
@patch("subprocess.Popen")
@patch.object(Path, "exists", return_value=True)
def test_mujoco_unified_launcher_script_success(
    mock_exists, mock_popen, mock_base_init
):
    launcher = MujocoUnifiedLauncher()
    launcher.show_error = MagicMock()

    launcher._launch_python_script("fake/path.py")
    mock_popen.assert_called_once()
    launcher.show_error.assert_not_called()


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
@patch.object(Path, "exists", return_value=False)
def test_mujoco_unified_launcher_script_not_found(mock_exists, mock_base_init):
    launcher = MujocoUnifiedLauncher()
    launcher.show_error = MagicMock()

    launcher._launch_python_script("fake/path.py")
    launcher.show_error.assert_called_once()


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
@patch("subprocess.Popen")
def test_mujoco_unified_launcher_module_success(mock_popen, mock_base_init):
    launcher = MujocoUnifiedLauncher()
    launcher.show_error = MagicMock()

    launcher._launch_python_module("my_module", "my/cwd")
    mock_popen.assert_called_once()
    launcher.show_error.assert_not_called()


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
@patch("subprocess.Popen", side_effect=OSError("Failed"))
def test_mujoco_unified_launcher_module_os_error(mock_popen, mock_base_init):
    launcher = MujocoUnifiedLauncher()
    launcher.show_error = MagicMock()

    launcher._launch_python_module("my_module", "my/cwd")
    launcher.show_error.assert_called_once()


def test_mujoco_unified_main():
    with patch("src.launchers.mujoco_unified_launcher.run_launcher") as mock_run:
        mock_run.return_value = 0
        assert mujoco_unified_main() == 0
        mock_run.assert_called_once_with(MujocoUnifiedLauncher)
