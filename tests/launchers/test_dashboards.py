"""Tests for various dashboard launchers."""

import sys
from unittest.mock import MagicMock, patch

from src.engines.physics_engines.drake.python.drake_physics_engine import (
    DrakePhysicsEngine,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
    MuJoCoPhysicsEngine,
)
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)
from src.launchers.drake_dashboard import main as drake_main
from src.launchers.matlab_launcher_unified import MatlabLauncher
from src.launchers.matlab_launcher_unified import main as matlab_main
from src.launchers.mujoco_dashboard import main as mujoco_main
from src.launchers.pinocchio_dashboard import main as pinocchio_main


def test_mujoco_dashboard_main():
    with patch("src.launchers.mujoco_dashboard.launch_dashboard") as mock_launch:
        mujoco_main()
        mock_launch.assert_called_once_with(
            engine_class=MuJoCoPhysicsEngine,
            title="MuJoCo Golf Analysis Dashboard (Unified)",
        )


def test_pinocchio_dashboard_main():
    with patch("src.launchers.pinocchio_dashboard.launch_dashboard") as mock_launch:
        pinocchio_main()
        mock_launch.assert_called_once_with(
            engine_class=PinocchioPhysicsEngine,
            title="Pinocchio Golf Analysis Dashboard",
        )


def test_drake_dashboard_main_no_args():
    # Mock sys.argv to just script name
    with (
        patch.object(sys, "argv", ["drake_dashboard.py"]),
        patch("src.launchers.drake_dashboard.get_qapp") as mock_qapp,
        patch("src.launchers.drake_dashboard.QFileDialog") as mock_dialog_class,
        patch("src.launchers.drake_dashboard.launch_dashboard") as mock_launch,
    ):
        mock_dialog = MagicMock()
        mock_dialog_class.return_value = mock_dialog
        mock_dialog.exec.return_value = True
        mock_dialog.selectedFiles.return_value = ["fake_model.urdf"]

        drake_main()

        mock_qapp.assert_called_once()
        mock_dialog.setNameFilter.assert_called_with("Model Files (*.urdf *.sdf *.xml)")
        mock_launch.assert_called_once_with(
            engine_class=DrakePhysicsEngine,
            title="Drake Golf Analysis Dashboard",
            model_path="fake_model.urdf",
        )


def test_drake_dashboard_main_with_args():
    with (
        patch.object(sys, "argv", ["drake_dashboard.py", "--model", "my_model.urdf"]),
        patch("src.launchers.drake_dashboard.launch_dashboard") as mock_launch,
    ):
        drake_main()
        mock_launch.assert_called_once_with(
            engine_class=DrakePhysicsEngine,
            title="Drake Golf Analysis Dashboard",
            model_path="my_model.urdf",
        )


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
def test_matlab_launcher(mock_base_init):
    launcher = MatlabLauncher()
    items = launcher.get_items()
    assert len(items) == 4

    # Check that they have required fields
    for item in items:
        assert item.name
        assert item.path
        assert item.item_type in ("model", "tool")


def test_matlab_main():
    with patch("src.launchers.matlab_launcher_unified.run_launcher") as mock_run:
        mock_run.return_value = 0
        assert matlab_main() == 0
        mock_run.assert_called_once_with(MatlabLauncher)
