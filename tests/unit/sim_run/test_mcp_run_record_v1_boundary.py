"""MCP submit_run/get_run must expose Fleet Agent API v1 run records (#15022)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sim_run.fleet_agent_api import CONTRACT_VERSION, assert_run_record_conformance
from src.sim_run.mcp import SimulationMcpServer
from src.sim_run.store import SimulationRunStore
from src.sim_run.worker import SimulationWorkerPool

pytestmark = pytest.mark.unit

_FLEET_STATUSES = frozenset({"QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "EXPIRED"})


def _payload(envelope: dict) -> dict:
    return json.loads(envelope["content"][0]["text"])


@pytest.fixture
def mcp_server(tmp_path: Path) -> SimulationMcpServer:
    store = SimulationRunStore(tmp_path / "runs")
    pool = SimulationWorkerPool(store, repo_root=Path(__file__).resolve().parents[3])
    return SimulationMcpServer(store=store, worker_pool=pool)


@pytest.mark.unit
def test_submit_run_returns_v1_fleet_status(mcp_server: SimulationMcpServer) -> None:
    envelope = mcp_server.tool_submit_run(
        {
            "engine_type": "pendulum",
            "duration": 0.05,
            "timestep": 0.01,
            "parameters": {"length": 1.0, "mass": 0.5},
        }
    )
    record = _payload(envelope)
    assert record["contract_version"] == CONTRACT_VERSION
    assert record["status"] in _FLEET_STATUSES
    assert record["status"] == "SUCCEEDED"
    assert not assert_run_record_conformance(record)


@pytest.mark.unit
def test_get_run_returns_v1_fleet_status(mcp_server: SimulationMcpServer) -> None:
    submit = _payload(
        mcp_server.tool_submit_run(
            {
                "engine_type": "pendulum",
                "duration": 0.05,
                "timestep": 0.01,
                "parameters": {"length": 1.0, "mass": 0.5},
            }
        )
    )
    run_id = submit["run_id"]
    record = _payload(mcp_server.tool_get_run({"run_id": run_id}))
    assert record["contract_version"] == CONTRACT_VERSION
    assert record["status"] == "SUCCEEDED"
    assert record["run_id"] == run_id
    assert not assert_run_record_conformance(record)
