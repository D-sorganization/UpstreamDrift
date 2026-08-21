"""API tests for the canonical attested longitudinal session contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import router

pytestmark = pytest.mark.integration

FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "launch_monitor"
    / "longitudinal_attested_v1.json"
)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _payload() -> dict[str, Any]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload.pop("expected")
    return payload


def test_contract_and_capabilities_publish_longitudinal_version(
    client: TestClient,
) -> None:
    contract = client.get(
        "/tools/launch-monitor-analytics/contracts/longitudinal-sessions/v1"
    )
    capabilities = client.get("/tools/launch-monitor-analytics/capabilities")

    assert contract.status_code == 200
    assert contract.json()["properties"]["contract_version"]["const"] == (
        "launch-monitor-longitudinal-session/1.0.0"
    )
    assert capabilities.status_code == 200
    assert capabilities.json()["longitudinal_session_contract_version"] == (
        "launch-monitor-longitudinal-session/1.0.0"
    )
    assert capabilities.json()["longitudinal_primary_unit"] == (
        "player_session_stratum"
    )
    assert capabilities.json()["longitudinal_causal_improvement"] is False


def test_api_returns_session_level_result_with_complete_backing(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/longitudinal-sessions",
        json=_payload(),
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "available"
    assert len(result["session_aggregates"]) == 12
    assert len(result["lineage"]["backing_records"]) == 36
    assert result["pooled_association"]["cluster_count"] == 4
    assert result["claims"]["causal_improvement"] is False


def test_api_returns_structured_unavailable_instead_of_pseudo_identity(
    client: TestClient,
) -> None:
    payload = _payload()
    payload["context"]["player_identity"] = {"trust_level": "not_provided"}

    response = client.post(
        "/tools/launch-monitor-analytics/v2/longitudinal-sessions",
        json=payload,
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "unavailable"
    assert result["availability"][0]["reason_code"] == "untrusted_player_identity"
    assert len(result["lineage"]["backing_records"]) == 36


def test_api_rejects_unknown_longitudinal_request_fields(client: TestClient) -> None:
    payload = _payload()
    payload["request"]["causal_improvement"] = True

    response = client.post(
        "/tools/launch-monitor-analytics/v2/longitudinal-sessions",
        json=payload,
    )

    assert response.status_code == 422
