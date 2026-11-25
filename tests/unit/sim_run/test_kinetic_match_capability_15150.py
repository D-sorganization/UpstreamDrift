"""Kinetic match over REST and MCP through one capability (#15150 D10)."""

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
from src.shared.python.motion_matching import kinetic_match as km
from src.sim_run import kinetic_match_capability as cap
from src.sim_run.mcp import SecurityPolicy, SimulationMcpServer
from src.sim_run.store import SimulationRunStore

pytestmark = pytest.mark.unit

_ROUTE = "/tools/agent-sim-run/kinetic-match"
_RECEIPT = {
    "method": km.METHOD,
    "verdict": "implausible",
    "reasons": ["root residual 80 N over limit"],
    "peaks": {"grip_force_n": 74.0, "joint_torque_n_m": 408.0},
}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimulationRunStore:
    runs = SimulationRunStore(tmp_path / "runs")
    monkeypatch.setattr(agent_sim_run, "_default_store", lambda: runs)
    return runs


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "b.npz").write_bytes(b"")
    monkeypatch.setenv("UD_KINETIC_MATCH_INPUT_ROOTS", str(tmp_path / "in"))
    monkeypatch.setenv("UD_KINETIC_MATCH_OUTPUT_ROOT", str(tmp_path / "out"))
    return tmp_path


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the heavy solve; record what the surfaces pass to it."""
    seen: list[dict[str, Any]] = []

    def fake(bundle: Path, out: Path, **kwargs: Any) -> dict[str, Any]:
        seen.append({"bundle": bundle, "out": out, **kwargs})
        return {
            "receipt": _RECEIPT,
            "out": out,
            "receipt_path": out.with_suffix(".kinetic_receipt.json"),
        }

    monkeypatch.setattr(km, "run_kinetic_match", fake)
    return seen


def _client(mode: str, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("GOLF_SUITE_MODE", mode)
    monkeypatch.setenv("GOLF_AUTH_DISABLED", "false")
    app = FastAPI()
    app.include_router(agent_sim_run.router)
    app.dependency_overrides[get_db_factory] = lambda: lambda: nullcontext(object())
    return TestClient(app)


def _mcp(store: SimulationRunStore, request: dict[str, Any]) -> Any:
    server = SimulationMcpServer(store=store, security_policy=SecurityPolicy())
    reply = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "kinetic_match", "arguments": {"request": request}},
        }
    )
    return json.loads(reply["result"]["content"][0]["text"])


def _both(store, monkeypatch, request) -> list[dict[str, Any]]:
    rest = _client("local", monkeypatch).post(_ROUTE, json=request)
    assert rest.status_code == 200
    return [rest.json(), _mcp(store, request)]


def test_rest_and_mcp_delegate_to_the_shared_kinetic_match(
    store, roots, calls, monkeypatch
) -> None:
    request = {"bundle": "b.npz", "out": "m/k.npz", "shaft_source": "reference"}
    records = _both(store, monkeypatch, request)
    assert len(calls) == 2
    for call in calls:
        assert call["bundle"] == (roots / "in" / "b.npz").resolve()
        assert call["out"] == (roots / "out" / "m" / "k.npz").resolve()
        assert call["shaft_source"] == "reference"
    rest, mcp = records
    assert rest["status"] == mcp["status"] == "succeeded"
    assert rest["delivery"] == mcp["delivery"]


def test_response_has_the_documented_fields(store, roots, calls, monkeypatch) -> None:
    record = _both(store, monkeypatch, {"bundle": "b.npz", "out": "k.npz"})[0]
    value = record["delivery"]["value"]
    assert set(cap.RESPONSE_FIELDS) <= set(value)
    assert value["verdict"] == "implausible"
    assert value["reasons"] == _RECEIPT["reasons"]
    assert value["peaks"] == _RECEIPT["peaks"]
    assert value["method"] == km.METHOD
    assert value["out_bundle"].endswith("k.npz")
    assert value["receipt_path"].endswith("k.kinetic_receipt.json")
    assert calls[0]["shaft_source"] is None


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ({"out": "k.npz"}, "bundle"),
        ({"bundle": "b.npz"}, "out"),
        ({"bundle": 3, "out": "k.npz"}, "non-empty string"),
        ({"bundle": "/etc/passwd", "out": "k.npz"}, "outside the allowed input"),
        ({"bundle": "b.npz", "out": "../k.npz"}, "outside the allowed output"),
        ({"bundle": "missing.npz", "out": "k.npz"}, "does not exist"),
        ({"bundle": "b.npz", "out": "k.txt"}, ".npz"),
        ({"bundle": "b.npz", "out": "k.npz", "shaft_source": "x"}, "shaft_source"),
        ({"bundle": "b.npz", "out": "k.npz", "stride": 2}, "unknown"),
    ],
)
def test_bad_requests_are_contract_errors_on_both_surfaces(
    request_body, message, store, roots, calls, monkeypatch
) -> None:
    for record in _both(store, monkeypatch, request_body):
        assert record["status"] == "failed"
        assert message in record["error"]["message"]
    assert calls == []


def test_remote_mode_rejects_unauthenticated_requests(store, monkeypatch) -> None:
    assert _client("remote", monkeypatch).post(_ROUTE, json={}).status_code == 401


def test_cli_delegates_to_the_shared_run(
    tmp_path: Path, calls: list[dict[str, Any]]
) -> None:
    assert km.main([str(tmp_path / "b.npz"), str(tmp_path / "k.npz")]) == 0
    assert calls[0]["bundle"] == tmp_path / "b.npz"
    assert calls[0]["settings_path"] == km.SETTINGS_PATH
