"""Native exporter schema over REST and MCP, one capability layer (#15150 G8)."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.database import get_db_factory
from src.api.routes import agent_sim_run
from src.sim_run.mcp import SecurityPolicy, SimulationMcpServer
from src.sim_run.store import SimulationRunStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimulationRunStore:
    runs = SimulationRunStore(tmp_path / "runs")
    monkeypatch.setattr(agent_sim_run, "_default_store", lambda: runs)
    return runs


def _client(mode: str, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("GOLF_SUITE_MODE", mode)
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "false")
    app = FastAPI()
    app.include_router(agent_sim_run.router)
    app.dependency_overrides[get_db_factory] = lambda: lambda: nullcontext(object())
    return TestClient(app)


def _mcp(store: SimulationRunStore, name: str, request: dict[str, Any]) -> Any:
    server = SimulationMcpServer(store=store, security_policy=SecurityPolicy())
    reply = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": {"request": request}},
        }
    )
    return json.loads(reply["result"]["content"][0]["text"])


def test_describe_is_identical_over_rest_and_mcp(
    store: SimulationRunStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = {"min_maturity": "beta"}
    rest = _client("local", monkeypatch).post(
        "/tools/agent-sim-run/native-export/describe", json=request
    )
    mcp = _mcp(store, "native_export_describe", request)
    assert rest.status_code == 200
    body = rest.json()
    assert body["status"] == mcp["status"] == "succeeded"
    assert body["input_sha256"] == mcp["input_sha256"]
    assert body["delivery"] == mcp["delivery"]
    assert body["delivery"]["value"]["schema"] == "ud-native-export-request/v1"


def test_invalid_request_is_a_failed_record_not_a_server_error(
    store: SimulationRunStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    rest = _client("local", monkeypatch).post(
        "/tools/agent-sim-run/native-export/validate",
        json={"options": {"bundle": "/etc/passwd", "out": "x", "swing": "s"}},
    )
    assert rest.status_code == 200
    assert rest.json()["status"] == "failed"
    assert "outside the allowed input roots" in rest.json()["error"]["message"]


@pytest.mark.parametrize("route", ["describe", "validate"])
def test_remote_mode_rejects_unauthenticated_requests(
    route: str, store: SimulationRunStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("remote", monkeypatch)
    reply = client.post(f"/tools/agent-sim-run/native-export/{route}", json={})
    assert reply.status_code == 401
