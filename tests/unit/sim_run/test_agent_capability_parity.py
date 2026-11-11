"""REST/MCP golden-case parity for Agent API capabilities (#15021)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import agent_sim_run
from src.sim_run.agent_capabilities import run_agent_capability
from src.sim_run.mcp import SimulationMcpServer
from src.sim_run.payload_bounds import canonical_json
from src.sim_run.store import SimulationRunStore
from src.sim_run.worker import SimulationWorkerPool

pytestmark = pytest.mark.unit


class _McpClient:
    def __init__(self, server: SimulationMcpServer) -> None:
        self.server = server
        self._id = 0

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._id += 1
        return self.server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": self._id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )


def _strip_volatile(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(record))
    for key in ("run_id", "created_at", "completed_at"):
        cleaned.pop(key, None)
    delivery = cleaned.get("delivery") or {}
    if delivery.get("encoding") == "artifact":
        ref = delivery.get("artifact_ref") or {}
        ref.pop("run_id", None)
    return cleaned


@pytest.fixture
def capability_store(tmp_path: Path) -> SimulationRunStore:
    return SimulationRunStore(tmp_path / "agent_runs")


def test_ztcf_golden_rest_and_mcp_match_bit_for_bit(
    capability_store: SimulationRunStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_sim_run, "_default_store", lambda: capability_store)
    app = FastAPI()
    app.include_router(agent_sim_run.router)
    client = TestClient(app)
    rest = client.post("/tools/agent-sim-run/counterfactual/ztcf-golden", json={}).json()

    pool = SimulationWorkerPool(capability_store, repo_root=tmp_path)
    mcp = _McpClient(SimulationMcpServer(store=capability_store, worker_pool=pool))
    mcp_raw = mcp.call("run_counterfactual_ztcf_golden", {})
    mcp_body = json.loads(mcp_raw["result"]["content"][0]["text"])

    assert rest["input_sha256"] == mcp_body["input_sha256"]
    assert canonical_json(_strip_volatile(rest)) == canonical_json(
        _strip_volatile(mcp_body)
    )
    banner = rest["delivery"]["value"]["qualification_banner"]
    assert "M^-1 tau" in banner["zvcf_definition"]
    assert banner["qualification_state"]["pointwise_only"] is True


def test_golf_shot_golden_rest_and_mcp_match_bit_for_bit(
    capability_store: SimulationRunStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_sim_run, "_default_store", lambda: capability_store)
    app = FastAPI()
    app.include_router(agent_sim_run.router)
    client = TestClient(app)
    rest = client.post("/tools/agent-sim-run/golf-simulator/golden-shot", json={}).json()

    pool = SimulationWorkerPool(capability_store, repo_root=tmp_path)
    mcp = _McpClient(SimulationMcpServer(store=capability_store, worker_pool=pool))
    mcp_raw = mcp.call("run_golf_simulator_golden_shot", {})
    mcp_body = json.loads(mcp_raw["result"]["content"][0]["text"])

    assert rest["input_sha256"] == mcp_body["input_sha256"]
    assert canonical_json(_strip_volatile(rest)) == canonical_json(
        _strip_volatile(mcp_body)
    )


def test_payload_over_64kib_returns_artifact(capability_store: SimulationRunStore) -> None:
    record = run_agent_capability(
        "golf_simulator_golden_shot",
        {"trajectory_padding": 4000},
        capability_store,
    )
    delivery = record["delivery"]
    assert delivery["encoding"] == "artifact"
    assert delivery["size_bytes"] > 64 * 1024
    ref = delivery["artifact_ref"]
    artifact_path = capability_store.get_run_dir(ref["run_id"]) / "artifacts" / ref[
        "artifact_name"
    ]
    assert artifact_path.is_file()
