"""Tests for the Drake engine __main__ module."""

import runpy
from unittest.mock import MagicMock, patch


@patch("src.shared.python.dashboard.launcher.launch_dashboard")
def test_drake_main(mock_launch_dashboard: MagicMock) -> None:
    """Test that the main module launches the dashboard correctly."""
    # Execute the __main__ module as if run via `python -m`
    runpy.run_module(
        "src.engines.physics_engines.drake.python.__main__", run_name="__main__"
    )

    # Verify the dashboard was launched with the correct engine class
    assert mock_launch_dashboard.call_count == 1
    args, kwargs = mock_launch_dashboard.call_args
    assert args[0].__name__ == "DrakePhysicsEngine"
    assert kwargs.get("title") == "Drake Physics Engine"
