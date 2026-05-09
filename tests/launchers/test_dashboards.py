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
