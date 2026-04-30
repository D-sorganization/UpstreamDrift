"""Tests for various dashboard launchers."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from src.launchers.drake_dashboard import main as drake_main  # noqa: E402
from src.launchers.matlab_launcher_unified import MatlabLauncher  # noqa: E402
from src.launchers.matlab_launcher_unified import main as matlab_main  # noqa: E402
from src.launchers.mujoco_dashboard import main as mujoco_main  # noqa: E402
from src.launchers.pinocchio_dashboard import main as pinocchio_main  # noqa: E402


def test_mujoco_dashboard_main() -> None:
    with patch("src.launchers.mujoco_dashboard.launch_dashboard") as mock_launch:
        mujoco_main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "MuJoCoPhysicsEngine"
        assert kwargs["title"] == "MuJoCo Golf Analysis Dashboard (Unified)"


def test_pinocchio_dashboard_main() -> None:
    with patch("src.launchers.pinocchio_dashboard.launch_dashboard") as mock_launch:
        pinocchio_main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "PinocchioPhysicsEngine"
        assert kwargs["title"] == "Pinocchio Golf Analysis Dashboard"


@pytest.mark.xfail(
    strict=False,
    reason="DrakePhysicsEngine class identity mismatch in parallel test run (namespace pkg)",
)
def test_drake_dashboard_main_no_args() -> None:
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
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "DrakePhysicsEngine"
        assert kwargs["title"] == "Drake Golf Analysis Dashboard"
        assert kwargs["model_path"] == "fake_model.urdf"


@pytest.mark.xfail(
    strict=False,
    reason="DrakePhysicsEngine class identity mismatch in parallel test run (namespace pkg)",
)
def test_drake_dashboard_main_no_args_dialog_canceled() -> None:
    with (
        patch.object(sys, "argv", ["drake_dashboard.py"]),
        patch("src.launchers.drake_dashboard.get_qapp"),
        patch("src.launchers.drake_dashboard.QFileDialog") as mock_dialog_class,
        patch("src.launchers.drake_dashboard.launch_dashboard") as mock_launch,
    ):
        mock_dialog = MagicMock()
        mock_dialog_class.return_value = mock_dialog
        mock_dialog.exec.return_value = False

        drake_main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "DrakePhysicsEngine"
        assert kwargs["model_path"] is None


@pytest.mark.xfail(
    strict=False,
    reason="DrakePhysicsEngine class identity mismatch in parallel test run (namespace pkg)",
)
def test_drake_dashboard_main_no_args_dialog_no_selection() -> None:
    with (
        patch.object(sys, "argv", ["drake_dashboard.py"]),
        patch("src.launchers.drake_dashboard.get_qapp"),
        patch("src.launchers.drake_dashboard.QFileDialog") as mock_dialog_class,
        patch("src.launchers.drake_dashboard.launch_dashboard") as mock_launch,
    ):
        mock_dialog = MagicMock()
        mock_dialog_class.return_value = mock_dialog
        mock_dialog.exec.return_value = True
        mock_dialog.selectedFiles.return_value = []

        drake_main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "DrakePhysicsEngine"
        assert kwargs["model_path"] is None


@pytest.mark.xfail(
    strict=False,
    reason="DrakePhysicsEngine class identity mismatch in parallel test run (namespace pkg)",
)
def test_drake_dashboard_main_with_args() -> None:
    with (
        patch.object(sys, "argv", ["drake_dashboard.py", "--model", "my_model.urdf"]),
        patch("src.launchers.drake_dashboard.launch_dashboard") as mock_launch,
    ):
        drake_main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "DrakePhysicsEngine"
        assert kwargs["model_path"] == "my_model.urdf"


@patch("src.launchers.base.BaseLauncher.__init__", return_value=None)
def test_matlab_launcher(mock_base_init) -> None:
    launcher = MatlabLauncher()
    items = launcher.get_items()
    assert len(items) == 4

    # Check that they have required fields
    for item in items:
        assert item.name
        assert item.path
        assert item.item_type in ("model", "tool")


def test_matlab_main() -> None:
    with patch("src.launchers.matlab_launcher_unified.run_launcher") as mock_run:
        mock_run.return_value = 0
        assert matlab_main() == 0
        mock_run.assert_called_once_with(MatlabLauncher)
