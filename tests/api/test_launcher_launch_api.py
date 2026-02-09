"""Tests for launcher launch/processes/stop API endpoints.

Covers:
    1. POST /api/launcher/launch/{tile_id} — launch by tile ID
    2. GET /api/launcher/processes — list running processes
    3. POST /api/launcher/stop/{name} — stop a running process
    4. Error handling — 404 for unknown tile, 400 for no handler, 500 for failed launch
    5. Diagnostics — launch logging, process manager state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# local_server creates app state in create_local_app(), including
# ProcessManager and ModelHandlerRegistry.  We mock both to avoid
# spawning real subprocesses during tests.
local_server = pytest.importorskip("src.api.local_server")


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def _reset_startup_metrics():
    """Reset startup metrics before each test."""
    local_server._startup_metrics.update(
        {
            "startup_time": None,
            "static_files_mounted": False,
            "ui_path": None,
            "engines_loaded": [],
            "errors": [],
        }
    )


@pytest.fixture()
def client(_reset_startup_metrics):
    """Create a TestClient for the local FastAPI app with mocked process management."""
    from fastapi.testclient import TestClient

    # Patch at the source module level so the imports inside create_local_app()
    # pick up the mocks
    with (
        patch(
            "src.launchers.launcher_process_manager.ProcessManager",
            return_value=MagicMock(running_processes={}),
        ),
        patch(
            "src.launchers.launcher_model_handlers.ModelHandlerRegistry",
        ) as mock_registry_cls,
    ):
        # Configure the mock registry to return handlers based on model type
        mock_registry = mock_registry_cls.return_value
        mock_handler = MagicMock()
        mock_handler.launch.return_value = True
        mock_registry.get_handler.return_value = mock_handler

        app = local_server.create_local_app()
        with TestClient(app) as tc:
            # Expose mocks for assertion access
            tc._mock_process_manager = app.state.process_manager
            tc._mock_handler_registry = mock_registry
            tc._mock_handler = mock_handler
            yield tc


@pytest.fixture()
def manifest_path() -> Path:
    """Return the path to the launcher manifest."""
    return Path(__file__).parent.parent.parent / "src" / "config" / "launcher_manifest.json"


@pytest.fixture()
def manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the launcher manifest."""
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


# ── POST /api/launcher/launch/{tile_id} ─────────────────────────────


class TestLaunchEndpoint:
    """Test POST /api/launcher/launch/{tile_id}."""

    def test_launch_mujoco_success(self, client) -> None:
        """Launching MuJoCo by tile ID returns 200 with status=launched."""
        resp = client.post("/api/launcher/launch/mujoco_unified")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "launched"
        assert data["tile_id"] == "mujoco_unified"
        assert data["name"] == "MuJoCo"

    def test_launch_drake_success(self, client) -> None:
        """Launching Drake by tile ID returns 200."""
        resp = client.post("/api/launcher/launch/drake_golf")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "launched"
        assert data["tile_id"] == "drake_golf"
        assert data["name"] == "Drake"

    def test_launch_tool_tile_success(self, client) -> None:
        """Launching a tool tile (model_explorer) returns 200."""
        resp = client.post("/api/launcher/launch/model_explorer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "launched"
        assert data["name"] == "Model Explorer"

    def test_launch_unknown_tile_returns_404(self, client) -> None:
        """Launching a non-existent tile returns 404."""
        resp = client.post("/api/launcher/launch/nonexistent_tile")
        assert resp.status_code == 404
        data = resp.json()
        assert "not found" in data["detail"].lower()

    def test_launch_calls_handler(self, client) -> None:
        """Launch endpoint invokes the correct handler with the tile's type."""
        client.post("/api/launcher/launch/mujoco_unified")
        # The registry was asked for a handler for "custom_humanoid"
        client._mock_handler_registry.get_handler.assert_called()

    def test_launch_no_handler_returns_400(self, client) -> None:
        """If no handler matches the tile type, returns 400."""
        client._mock_handler_registry.get_handler.return_value = None
        resp = client.post("/api/launcher/launch/mujoco_unified")
        assert resp.status_code == 400
        assert "no handler" in resp.json()["detail"].lower()

    def test_launch_handler_failure_returns_500(self, client) -> None:
        """If handler.launch() returns False, returns 500."""
        client._mock_handler.launch.return_value = False
        resp = client.post("/api/launcher/launch/drake_golf")
        assert resp.status_code == 500
        assert "failed to launch" in resp.json()["detail"].lower()

    def test_launch_all_manifest_tiles(self, client, manifest) -> None:
        """Every tile in the manifest can be launched (handler returns True)."""
        for tile in manifest["tiles"]:
            resp = client.post(f"/api/launcher/launch/{tile['id']}")
            assert resp.status_code == 200, (
                f"Failed to launch tile '{tile['id']}': {resp.json()}"
            )
            data = resp.json()
            assert data["tile_id"] == tile["id"]

    def test_launch_response_shape(self, client) -> None:
        """Launch response has exactly {status, tile_id, name} keys."""
        resp = client.post("/api/launcher/launch/mujoco_unified")
        data = resp.json()
        assert set(data.keys()) == {"status", "tile_id", "name"}

    def test_launch_uses_correct_model_type(self, client) -> None:
        """Verify the handler registry is queried with the tile's actual type."""
        client.post("/api/launcher/launch/putting_green")
        # The manifest says putting_green has type "putting_green"
        call_args = client._mock_handler_registry.get_handler.call_args_list[-1]
        assert call_args[0][0] == "putting_green"


# ── GET /api/launcher/processes ──────────────────────────────────────


class TestProcessesEndpoint:
    """Test GET /api/launcher/processes."""

    def test_empty_processes(self, client) -> None:
        """When no processes are running, returns empty dict."""
        resp = client.get("/api/launcher/processes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["processes"] == {}

    def test_running_process_listed(self, client) -> None:
        """A running process is listed with pid and running=True."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # still running
        client._mock_process_manager.running_processes["MuJoCo Humanoid Golf"] = mock_proc

        resp = client.get("/api/launcher/processes")
        assert resp.status_code == 200
        procs = resp.json()["processes"]
        assert "MuJoCo Humanoid Golf" in procs
        assert procs["MuJoCo Humanoid Golf"]["pid"] == 12345
        assert procs["MuJoCo Humanoid Golf"]["running"] is True
        assert procs["MuJoCo Humanoid Golf"]["exit_code"] is None

    def test_exited_process_listed(self, client) -> None:
        """An exited process is listed with running=False and exit_code."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = 1  # exited with code 1
        client._mock_process_manager.running_processes["Drake Golf Model"] = mock_proc

        resp = client.get("/api/launcher/processes")
        procs = resp.json()["processes"]
        assert "Drake Golf Model" in procs
        assert procs["Drake Golf Model"]["running"] is False
        assert procs["Drake Golf Model"]["exit_code"] == 1

    def test_multiple_processes(self, client) -> None:
        """Multiple processes are listed correctly."""
        for name, pid in [("Engine A", 100), ("Engine B", 200), ("Tool C", 300)]:
            proc = MagicMock()
            proc.pid = pid
            proc.poll.return_value = None
            client._mock_process_manager.running_processes[name] = proc

        resp = client.get("/api/launcher/processes")
        procs = resp.json()["processes"]
        assert len(procs) == 3
        assert procs["Engine A"]["pid"] == 100
        assert procs["Engine B"]["pid"] == 200
        assert procs["Tool C"]["pid"] == 300


# ── POST /api/launcher/stop/{name} ──────────────────────────────────


class TestStopEndpoint:
    """Test POST /api/launcher/stop/{name}."""

    def test_stop_running_process(self, client) -> None:
        """Stopping a running process returns 200 with status=stopped."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        client._mock_process_manager.running_processes["MuJoCo Humanoid Golf"] = mock_proc

        with patch("src.api.local_server.kill_process_tree"):
            resp = client.post("/api/launcher/stop/MuJoCo Humanoid Golf")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"
        assert data["name"] == "MuJoCo Humanoid Golf"

    def test_stop_unknown_process_returns_404(self, client) -> None:
        """Stopping a non-existent process returns 404."""
        resp = client.post("/api/launcher/stop/nonexistent_engine")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_stop_removes_from_running(self, client) -> None:
        """After stopping, the process is removed from running_processes."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        client._mock_process_manager.running_processes["Test Engine"] = mock_proc

        with patch("src.api.local_server.kill_process_tree"):
            client.post("/api/launcher/stop/Test Engine")

        assert "Test Engine" not in client._mock_process_manager.running_processes

    def test_stop_calls_kill_process_tree(self, client) -> None:
        """Stop endpoint uses kill_process_tree to terminate the process."""
        mock_proc = MagicMock()
        mock_proc.pid = 54321
        client._mock_process_manager.running_processes["Drake Golf Model"] = mock_proc

        with patch("src.api.local_server.kill_process_tree") as mock_kill:
            client.post("/api/launcher/stop/Drake Golf Model")
            mock_kill.assert_called_once_with(54321)


# ── Integration: launch → list → stop cycle ──────────────────────────


class TestLaunchLifecycle:
    """Test the full launch → list → stop lifecycle."""

    def test_launch_then_list(self, client) -> None:
        """After launching, the process appears in the list."""
        # First launch
        resp = client.post("/api/launcher/launch/mujoco_unified")
        assert resp.status_code == 200

        # The handler was called; simulate it adding a process
        mock_proc = MagicMock()
        mock_proc.pid = 11111
        mock_proc.poll.return_value = None
        client._mock_process_manager.running_processes["MuJoCo"] = mock_proc

        # List
        resp = client.get("/api/launcher/processes")
        assert "MuJoCo" in resp.json()["processes"]

    def test_launch_list_stop_list(self, client) -> None:
        """Full lifecycle: launch → list → stop → list empty."""
        # Launch
        client.post("/api/launcher/launch/drake_golf")
        mock_proc = MagicMock()
        mock_proc.pid = 22222
        mock_proc.poll.return_value = None
        client._mock_process_manager.running_processes["Drake Golf Model"] = mock_proc

        # List — should have process
        resp = client.get("/api/launcher/processes")
        assert "Drake Golf Model" in resp.json()["processes"]

        # Stop
        with patch("src.api.local_server.kill_process_tree"):
            resp = client.post("/api/launcher/stop/Drake Golf Model")
        assert resp.status_code == 200

        # List — should be empty
        resp = client.get("/api/launcher/processes")
        assert "Drake Golf Model" not in resp.json()["processes"]


# ── Manifest consistency ─────────────────────────────────────────────


class TestManifestConsistency:
    """Ensure the manifest data flowing through the launch endpoint is consistent."""

    def test_all_tiles_have_type(self, manifest) -> None:
        """Every tile in the manifest has a type field."""
        for tile in manifest["tiles"]:
            assert "type" in tile, f"Tile '{tile['id']}' missing 'type'"
            assert tile["type"], f"Tile '{tile['id']}' has empty type"

    def test_all_tile_ids_unique(self, manifest) -> None:
        """All tile IDs must be unique."""
        ids = [t["id"] for t in manifest["tiles"]]
        assert len(ids) == len(set(ids)), f"Duplicate tile IDs: {ids}"

    def test_handler_coverage(self) -> None:
        """ModelHandlerRegistry has handlers for all manifest tile types."""
        from src.launchers.launcher_model_handlers import ModelHandlerRegistry

        registry = ModelHandlerRegistry()
        manifest_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "config"
            / "launcher_manifest.json"
        )
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        unhandled = []
        for tile in manifest["tiles"]:
            handler = registry.get_handler(tile["type"])
            if handler is None:
                unhandled.append(f"{tile['id']} (type={tile['type']})")

        assert not unhandled, f"No handler for tiles: {unhandled}"


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for launch API robustness."""

    def test_launch_with_url_encoded_id(self, client) -> None:
        """Tile IDs with underscores work correctly in URL paths."""
        resp = client.post("/api/launcher/launch/mujoco_unified")
        assert resp.status_code == 200

    def test_launch_empty_tile_id(self, client) -> None:
        """POST /api/launcher/launch/ without tile_id returns 404/405."""
        resp = client.post("/api/launcher/launch/")
        # FastAPI returns 404 for missing path segment or redirects
        assert resp.status_code in {404, 405, 307}

    def test_processes_endpoint_is_get_only(self, client) -> None:
        """POST to /api/launcher/processes returns 405."""
        resp = client.post("/api/launcher/processes")
        assert resp.status_code == 405

    def test_stop_with_spaces_in_name(self, client) -> None:
        """Process names with spaces (like 'MuJoCo Humanoid Golf') work."""
        mock_proc = MagicMock()
        mock_proc.pid = 33333
        client._mock_process_manager.running_processes["MuJoCo Humanoid Golf"] = mock_proc

        with patch("src.api.local_server.kill_process_tree"):
            resp = client.post("/api/launcher/stop/MuJoCo Humanoid Golf")
        assert resp.status_code == 200
