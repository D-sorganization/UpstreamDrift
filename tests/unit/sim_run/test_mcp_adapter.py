"""Unit tests and falsification harness for simulation MCP adapter (UD #14092 / #14085)."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.sim_run.mcp import SecurityPolicy, SimulationMcpServer
from src.sim_run.store import SimulationRunStore
from src.sim_run.worker import SimulationWorkerPool


class InProcessMcpClient:
    """Synchronous in-process test client for SimulationMcpServer."""

    def __init__(self, server: SimulationMcpServer) -> None:
        self.server = server
        self._id = 0

    def tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self._id += 1
        return self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": self._id,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }
        )


@pytest.fixture
def sim_service(
    tmp_path: Path,
) -> tuple[SimulationRunStore, SimulationWorkerPool, Path]:
    store = SimulationRunStore(tmp_path / "sim_runs")
    pool = SimulationWorkerPool(store, repo_root=Path(__file__).resolve().parents[3])
    return store, pool, tmp_path / "sim_runs"


@pytest.mark.unit
def test_mcp_client_round_trip_all_four_tools(
    sim_service: tuple[SimulationRunStore, SimulationWorkerPool, Path],
) -> None:
    """Falsification test 1: in-process client round-trip succeeds for all 4 tools without sockets."""
    store, pool, _ = sim_service
    client = InProcessMcpClient(SimulationMcpServer(store=store, worker_pool=pool))

    with patch.object(socket.socket, "bind") as mock_bind:
        init_res = client.server.handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        assert (
            init_res.get("result", {}).get("serverInfo", {}).get("name") == "ud-sim-mcp"
        )

        tools = client.server.handle_request(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )["result"]["tools"]
        assert {"describe_engines", "submit_run", "get_run", "fetch_artifact"}.issubset(
            {t["name"] for t in tools}
        )

        desc_resp = client.tool("describe_engines", {})
        desc_data = json.loads(desc_resp["result"]["content"][0]["text"])
        assert any(e["engine_type"] == "pendulum" for e in desc_data["engines"])

        submit_resp = client.tool(
            "submit_run",
            {
                "engine_type": "pendulum",
                "duration": 0.1,
                "timestep": 0.01,
                "parameters": {"length": 1.0, "mass": 0.5},
            },
        )
        assert "result" in submit_resp, f"Submit failed: {submit_resp}"
        run_record = json.loads(submit_resp["result"]["content"][0]["text"])
        run_id = run_record["run_id"]
        assert run_record["status"] == "succeeded", (
            f"Run failed with: {run_record.get('error')}"
        )

        get_resp = client.tool("get_run", {"run_id": run_id})
        assert (
            json.loads(get_resp["result"]["content"][0]["text"])["status"]
            == "succeeded"
        )

        store.save_artifact(run_id, "summary.json", '{"benchmark": "pass"}')
        art_resp = client.tool(
            "fetch_artifact", {"run_id": run_id, "artifact_name": "summary.json"}
        )
        assert (
            "benchmark"
            in json.loads(art_resp["result"]["content"][0]["text"])["content"]
        )

        mock_bind.assert_not_called()


@pytest.mark.unit
def test_mcp_private_mode_refuses_execution(
    sim_service: tuple[SimulationRunStore, SimulationWorkerPool, Path],
) -> None:
    """Falsification test 2: Private Mode actively refuses simulation submission at boundary."""
    store, pool, _ = sim_service
    server = SimulationMcpServer(
        store=store, worker_pool=pool, security_policy=SecurityPolicy(private_mode=True)
    )
    resp = InProcessMcpClient(server).tool("submit_run", {"engine_type": "pendulum"})
    assert resp.get("error", {}).get("code") == -32000
    assert "Private Mode active" in resp["error"]["message"]


@pytest.mark.unit
def test_mcp_auth_token_enforcement(
    sim_service: tuple[SimulationRunStore, SimulationWorkerPool, Path],
) -> None:
    """Boundary test: verify auth token requirement rejects unauthenticated requests."""
    store, pool, _ = sim_service
    srv = SimulationMcpServer(
        store=store,
        worker_pool=pool,
        security_policy=SecurityPolicy(auth_token="token"),
    )
    c = InProcessMcpClient(srv)
    assert "Unauthorized" in c.tool("describe_engines", {}).get("error", {}).get(
        "message", ""
    )
    assert "Unauthorized" in c.tool("describe_engines", {"auth_token": "x"}).get(
        "error", {}
    ).get("message", "")
    assert "result" in c.tool("describe_engines", {"auth_token": "token"})


@pytest.mark.unit
def test_mcp_artifact_path_traversal_defense(
    sim_service: tuple[SimulationRunStore, SimulationWorkerPool, Path],
) -> None:
    """Safety test: verify path traversal in artifact_name is strictly rejected."""
    store, pool, _ = sim_service
    client = InProcessMcpClient(SimulationMcpServer(store=store, worker_pool=pool))
    resp = client.tool(
        "fetch_artifact", {"run_id": "r1", "artifact_name": "../../etc/passwd"}
    )
    assert "path traversal" in resp.get("error", {}).get("message", "")
