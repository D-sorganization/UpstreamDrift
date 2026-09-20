"""Tests for MeshCat and Gepetto viewer lifecycle management (MV-05 #10481)."""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.motion_matching.viewer_lifecycle import (
    PortCollisionError,
    ViewerEndpoint,
    ViewerProcessManager,
    is_port_in_use,
    wait_for_port,
)


pytestmark = pytest.mark.unit


def _find_free_port() -> int:
    """Find an unbound TCP port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_is_port_in_use_and_wait_for_port():
    port = _find_free_port()
    assert not is_port_in_use("127.0.0.1", port)
    assert not wait_for_port("127.0.0.1", port, timeout_s=0.1, interval_s=0.02)

    # Bind and listen on port
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", port))
        server.listen(128)
        assert is_port_in_use("127.0.0.1", port)
        assert wait_for_port("127.0.0.1", port, timeout_s=0.5, interval_s=0.02)
    finally:
        server.close()


def test_viewer_process_manager_starts_and_stops_owned_process():
    manager = ViewerProcessManager()
    dummy_cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    port = _find_free_port()

    # Simulate start of server
    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "src.shared.python.motion_matching.viewer_lifecycle.wait_for_port",
            return_value=True,
        ),
    ):
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        endpoint = manager.start_server(
            name="test_server",
            cmd=dummy_cmd,
            host="127.0.0.1",
            port=port,
            timeout_s=1.0,
        )

        assert endpoint.name == "test_server"
        assert endpoint.host == "127.0.0.1"
        assert endpoint.port == port
        assert endpoint.pid == 99999
        assert manager.is_owned(endpoint.pid)

        # Stop server
        stopped = manager.stop_server(endpoint)
        assert stopped
        mock_proc.terminate.assert_called_once()
        assert not manager.is_owned(endpoint.pid)


def test_viewer_process_manager_never_terminates_unowned_process():
    manager = ViewerProcessManager()
    unowned_endpoint = ViewerEndpoint(
        name="unowned_server",
        host="127.0.0.1",
        port=12321,
        pid=12345,
        is_owned=False,
    )

    stopped = manager.stop_server(unowned_endpoint)
    assert not stopped


def test_viewer_process_manager_detects_existing_listener_without_colliding():
    port = _find_free_port()
    manager = ViewerProcessManager()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", port))
        server.listen(128)

        # Attempting to start with reuse_existing=True discovers existing endpoint
        endpoint = manager.start_server(
            name="existing_gepetto",
            cmd=["gepetto-gui"],
            host="127.0.0.1",
            port=port,
            reuse_existing=True,
        )
        assert endpoint.port == port
        assert not endpoint.is_owned

        # Attempting to start with reuse_existing=False raises PortCollisionError
        with pytest.raises(PortCollisionError, match="already in use"):
            manager.start_server(
                name="colliding_gepetto",
                cmd=["gepetto-gui"],
                host="127.0.0.1",
                port=port,
                reuse_existing=False,
            )
    finally:
        server.close()


def test_viewer_process_manager_handles_crash_on_startup():
    manager = ViewerProcessManager()
    port = _find_free_port()

    with (
        patch("subprocess.Popen") as mock_popen,
        patch(
            "src.shared.python.motion_matching.viewer_lifecycle.wait_for_port",
            return_value=False,
        ),
    ):
        mock_proc = MagicMock()
        mock_proc.pid = 88888
        mock_proc.poll.return_value = 1  # Exited immediately with error
        mock_proc.communicate.return_value = ("", "Error: cannot open display")
        mock_popen.return_value = mock_proc

        with pytest.raises(RuntimeError, match="exited unexpectedly.*display"):
            manager.start_server(
                name="crashing_server",
                cmd=["crash_binary"],
                host="127.0.0.1",
                port=port,
                timeout_s=0.2,
            )


def test_viewer_process_manager_meshcat_dynamic_url():
    manager = ViewerProcessManager()
    # Test URL parsing logic for MeshCat dynamic ports (7000, 7001, etc.)
    url_7000 = manager.resolve_meshcat_url("http://127.0.0.1:7000/static/")
    assert url_7000 == "http://127.0.0.1:7000/static/"

    url_7001 = manager.resolve_meshcat_url("http://localhost:7001/static/")
    assert url_7001 == "http://localhost:7001/static/"

    with pytest.raises(ValueError, match="Invalid MeshCat URL"):
        manager.resolve_meshcat_url("not-a-url")
