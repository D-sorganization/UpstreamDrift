"""Tests that the browser opening timer is daemonized in local_server.py (#6924)."""

from __future__ import annotations

import threading
from unittest.mock import patch

from src.api.local_server import main


def test_local_server_timer_daemon() -> None:
    """The browser opening Timer in local_server.py must be set to daemon=True."""

    # Store instances of Timer that get created
    created_timers: list[threading.Timer] = []
    original_timer = threading.Timer

    def mock_timer_class(interval, function, args=None, kwargs=None):
        timer = original_timer(interval, function, args, kwargs)
        created_timers.append(timer)
        return timer

    with (
        patch("threading.Timer", side_effect=mock_timer_class),
        patch("uvicorn.run") as mock_run,
        patch("src.shared.python.config.environment.get_golf_port", return_value=8000),
        patch("src.api.local_server.print_matrix_status"),
        patch("src.api.local_server.print_server_info"),
        patch("src.api.local_server.print_logo_animated"),
        patch("webbrowser.open"),
    ):
        main()

        # Verify uvicorn.run was called
        mock_run.assert_called_once()

        # Verify a Timer was created and started
        assert len(created_timers) == 1
        timer = created_timers[0]
        assert timer.daemon is True

        # Clean up timer so it doesn't execute function in background
        timer.cancel()
