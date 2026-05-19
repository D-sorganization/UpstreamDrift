"""Tests for various dashboard launchers."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from src.launchers.drake_dashboard import main as drake_main  # noqa: E402
from src.launchers.mujoco_dashboard import main as mujoco_main  # noqa: E402
from src.launchers.pinocchio_dashboard import main as pinocchio_main  # noqa: E402


def test_mujoco_dashboard_main() -> None:
    pytest.importorskip("mujoco")
    with patch("src.launchers.mujoco_dashboard.launch_dashboard") as mock_launch:
        mujoco_main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "MuJoCoPhysicsEngine"
        assert kwargs["title"] == "MuJoCo Golf Analysis Dashboard (Unified)"


def test_pinocchio_dashboard_main() -> None:
    pytest.importorskip("pinocchio")
    with patch("src.launchers.pinocchio_dashboard.launch_dashboard") as mock_launch:
        pinocchio_main()
        mock_launch.assert_called_once()
        _, kwargs = mock_launch.call_args
        assert kwargs["engine_class"].__name__ == "PinocchioPhysicsEngine"
        assert kwargs["title"] == "Pinocchio Golf Analysis Dashboard"
