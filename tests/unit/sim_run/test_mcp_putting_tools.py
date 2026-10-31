"""Putting MCP tool tests on the shared stdio adapter (epic #14141 step 9a).

Reuses ``test_mcp_adapter.py``'s in-process JSON-RPC client; assertions
are invariants/parity with ``putting_dynamics``, never output snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.putting_dynamics import HOLE_RADIUS_M, capture_speed_mps
from src.sim_run.mcp import SecurityPolicy, SimulationMcpServer
from src.sim_run.store import SimulationRunStore
from src.sim_run.worker import SimulationWorkerPool
from tests.unit.sim_run.test_mcp_adapter import InProcessMcpClient

_PUTTING_TOOL_NAMES = {
    "putting_simulate_putt",
    "putting_read_green",
    "putting_stimpmeter",
    "putting_capture_speed",
}
#: Short putt used throughout so `solve_aim`'s bisection stays fast.
_SHORT_DISTANCE_M = 1.5


@pytest.fixture
def client(tmp_path: Path) -> InProcessMcpClient:
    store = SimulationRunStore(tmp_path / "sim_runs")
    pool = SimulationWorkerPool(store, repo_root=Path(__file__).resolve().parents[3])
    return InProcessMcpClient(SimulationMcpServer(store=store, worker_pool=pool))


def _result_payload(response: dict[str, Any]) -> dict[str, Any]:
    text: str = response["result"]["content"][0]["text"]
    return json.loads(text)


@pytest.mark.unit
def test_tools_list_includes_putting_tools_with_schemas(
    client: InProcessMcpClient,
) -> None:
    tools = client.server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )["result"]["tools"]
    by_name = {t["name"]: t for t in tools}
    assert _PUTTING_TOOL_NAMES.issubset(by_name)
    for name in _PUTTING_TOOL_NAMES:
        schema = by_name[name]["inputSchema"]
        assert schema["type"] == "object"
        assert "auth_token" in schema["properties"]


@pytest.mark.unit
def test_simulate_putt_round_trip_reports_roll_model(
    client: InProcessMcpClient,
) -> None:
    resp = client.tool(
        "putting_simulate_putt",
        {"putter_speed_mps": 1.0, "distance_m": _SHORT_DISTANCE_M},
    )
    assert "result" in resp, resp
    payload = _result_payload(resp)
    assert payload["roll_model"] == "usga-stimp-roll/1"
    assert payload["total_distance_m"] >= 0.0
    assert payload["time_s"] >= 0.0


@pytest.mark.unit
def test_read_green_on_flat_green_holes_straight(client: InProcessMcpClient) -> None:
    resp = client.tool("putting_read_green", {"distance_m": _SHORT_DISTANCE_M})
    assert "result" in resp, resp
    payload = _result_payload(resp)
    assert payload["roll_model"] == "usga-stimp-roll/1"
    assert payload["heading_deg"] == pytest.approx(0.0, abs=1.0)
    assert payload["holed"] is True


@pytest.mark.unit
def test_stimpmeter_round_trip(client: InProcessMcpClient) -> None:
    resp = client.tool("putting_stimpmeter", {"stimp_ft": 10.0})
    assert "result" in resp, resp
    payload = _result_payload(resp)
    assert payload["measured_stimp_ft"] > 0.0


@pytest.mark.unit
def test_capture_speed_at_zero_offset_matches_shared_core(
    client: InProcessMcpClient,
) -> None:
    resp = client.tool("putting_capture_speed", {})
    assert "result" in resp, resp
    payload = _result_payload(resp)
    assert payload["max_capture_speed_mps"] == pytest.approx(capture_speed_mps())


@pytest.mark.unit
def test_capture_speed_falls_with_offset(client: InProcessMcpClient) -> None:
    centre = _result_payload(client.tool("putting_capture_speed", {}))
    off_axis = _result_payload(
        client.tool("putting_capture_speed", {"entry_offset_m": 0.5 * HOLE_RADIUS_M})
    )
    assert off_axis["max_capture_speed_mps"] < centre["max_capture_speed_mps"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,args",
    [
        ("putting_simulate_putt", {"putter_speed_mps": "fast", "distance_m": 1.5}),
        ("putting_simulate_putt", {"putter_speed_mps": 1.0, "distance_m": -1.0}),
        ("putting_read_green", {"distance_m": 0.0}),
        ("putting_stimpmeter", {"stimp_ft": 100.0}),
        ("putting_capture_speed", {"entry_offset_m": HOLE_RADIUS_M * 2}),
    ],
)
def test_invalid_arguments_return_invalid_params_error(
    client: InProcessMcpClient, name: str, args: dict[str, Any]
) -> None:
    resp = client.tool(name, args)
    assert resp.get("error", {}).get("code") == -32602, resp


@pytest.mark.unit
def test_auth_refusal_when_security_policy_requires_token(tmp_path: Path) -> None:
    store = SimulationRunStore(tmp_path / "sim_runs")
    pool = SimulationWorkerPool(store, repo_root=Path(__file__).resolve().parents[3])
    server = SimulationMcpServer(
        store=store,
        worker_pool=pool,
        security_policy=SecurityPolicy(auth_token="secret"),
    )
    authed_client = InProcessMcpClient(server)
    refused = authed_client.tool("putting_capture_speed", {"auth_token": "wrong"})
    assert "Unauthorized" in refused.get("error", {}).get("message", "")
    accepted = authed_client.tool("putting_capture_speed", {"auth_token": "secret"})
    assert "result" in accepted, accepted
