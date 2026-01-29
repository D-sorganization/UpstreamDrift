"""Integration tests for the application launcher and engine detection."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from PyQt6 import QtWidgets

from src.shared.python import engine_availability
from src.shared.python.dashboard import launcher
from src.shared.python.environment import is_docker, is_production, is_wsl


@pytest.fixture(scope="session")
def qapp():
    """Fixture to ensure a QApplication exists."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    return app


def test_environment_detection():
    """Test environment detection functions."""
    # These depend on the actual env, but we can verify they run without error
    # and return booleans
    assert isinstance(is_docker(), bool)
    assert isinstance(is_wsl(), bool)
    assert isinstance(is_production(), bool)


def test_engine_availability_check():
    """Test that we can check for physics engines."""
    # We expect at least one standard library to be available (numpy)
    assert engine_availability.NUMPY_AVAILABLE is True

    # Check is_engine_available function
    assert engine_availability.is_engine_available("numpy") is True
    assert engine_availability.is_engine_available("non_existent_engine") is False


@patch("src.shared.python.dashboard.launcher.UnifiedDashboardWindow")
@patch("src.shared.python.dashboard.launcher.sys.exit")
@patch("src.shared.python.dashboard.launcher.get_qapp")
def test_dashboard_launch(mock_get_qapp, mock_exit, mock_window, qapp):
    """Test launching the dashboard with a mock engine."""

    # Mock valid engine class
    mock_engine_class = MagicMock()
    mock_engine = MagicMock()
    mock_engine_class.return_value = mock_engine

    # Mock QApp
    mock_get_qapp.return_value = qapp

    # Run launch
    launcher.launch_dashboard(
        mock_engine_class,
        title="Test Dashboard",
        engine_args=["arg1"],
        engine_kwargs={"kwarg1": "val"},
    )

    # Verify initialization
    mock_engine_class.assert_called_once_with("arg1", kwarg1="val")
    mock_window.assert_called_once_with(mock_engine, title="Test Dashboard")
    mock_exit.assert_called_once()


def test_mujoco_availability_logic():
    """Verify MuJoCo logic (ensure it's not strictly disabled by default logic anymore)."""
    # This test verifies that checking for mujoco doesn't raise an error
    # even if it returns False.
    try:
        available = engine_availability.is_engine_available("mujoco")
        assert isinstance(available, bool)
    except Exception as e:
        pytest.fail(f"Checking MuJoCo availability raised exception: {e}")
