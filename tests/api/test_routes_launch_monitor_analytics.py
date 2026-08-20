"""Tests for the UI-neutral launch-monitor analytics API contract."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import router

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _records() -> list[dict[str, object]]:
    return [
        {
            "shot_id": f"shot-{index}",
            "session_id": "api-session",
            "monitor_vendor": "FlightScope",
            "club_speed": float(index),
            "ball_speed": 1.5 * index + 2.0,
        }
        for index in range(1, 31)
    ]


def test_capabilities_are_machine_readable_and_versioned(client: TestClient) -> None:
    response = client.get("/tools/launch-monitor-analytics/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "1.0.0"
    assert payload["analysis_modes"] == ["correlation", "regression", "comprehensive"]
    assert payload["aggregate_regression_allowed"] is False
    assert payload["supported_contract_versions"] == ["1.0.0", "2.0.0"]


def test_v2_schema_is_published_from_the_canonical_python_model(
    client: TestClient,
) -> None:
    response = client.get("/tools/launch-monitor-analytics/contracts/v2")
    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "LaunchMonitorAnalysisResultV2"
    assert payload["properties"]["contract_version"]["const"] == "2.0.0"


def test_v2_result_is_registered_in_openapi(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    operation = schema["paths"]["/tools/launch-monitor-analytics/v2/analyze"]["post"]
    result_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    assert result_schema["$ref"].endswith("/LaunchMonitorAnalysisResultV2")


def test_analyze_accepts_inline_records_and_returns_provenance(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/analyze",
        json={
            "records": _records(),
            "analysis": {
                "outcome": "ball_speed",
                "predictors": ["club_speed"],
                "analysis_mode": "comprehensive",
                "min_samples": 10,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "1.0.0"
    assert payload["result"]["dataset"]["monitor_vendors"] == ["FlightScope"]
    assert len(payload["result"]["dataset"]["fingerprint_sha256"]) == 64
    assert payload["result"]["regression"]["r_squared"] == pytest.approx(1.0)


def test_v2_analyze_returns_traceable_contract_without_changing_v1(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/analyze",
        json={
            "records": _records(),
            "analysis": {
                "outcome": "ball_speed",
                "predictors": ["club_speed"],
                "analysis_mode": "comprehensive",
                "min_samples": 10,
            },
            "context": {
                "authority": {
                    "dataset_id": "api-fixture",
                    "repository": "D-sorganization/UpstreamDrift",
                    "commit": "4b898d237" + "0" * 31,
                },
                "player_identity": {"trust_level": "not_provided"},
                "sources": [
                    {
                        "source_id": "fixture-records",
                        "file_sha256": "b" * 64,
                        "session_ids": ["api-session"],
                        "rights_status": "public_redistributable",
                    }
                ],
            },
            "model_provenance": [
                {
                    "model_id": "penner-flight",
                    "version": "1.0.0",
                    "code_commit": "a" * 40,
                    "relationship_to_vendor": "independent_physics",
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "2.0.0"
    assert payload["status"] == "available"
    assert payload["lineage"]["authority"]["dataset_id"] == "api-fixture"
    assert payload["lineage"]["sources"][0]["source_id"] == "fixture-records"
    assert len(payload["lineage"]["backing_records"]) == 30
    assert payload["units"]["ball_speed"]["canonical_unit"] == "m/s"
    assert payload["model_provenance"][0]["code_commit"] == "a" * 40


def test_v2_analyze_reports_missing_selected_column_as_contract_error(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/analyze",
        json={
            "records": _records(),
            "analysis": {
                "outcome": "ball_speed",
                "predictors": ["missing_metric"],
                "analysis_mode": "correlation",
            },
        },
    )
    assert response.status_code == 400
    assert "Columns not present" in response.json()["detail"]
    assert "missing_metric" in response.json()["detail"]


def test_analyze_rejects_aggregate_regression(client: TestClient) -> None:
    records = _records()
    for record in records:
        record["observation_kind"] = "aggregate"
    response = client.post(
        "/tools/launch-monitor-analytics/analyze",
        json={
            "records": records,
            "analysis": {
                "outcome": "ball_speed",
                "predictors": ["club_speed"],
                "analysis_mode": "regression",
                "allow_aggregate": True,
            },
        },
    )
    assert response.status_code == 400
    assert "Aggregate observations cannot enter regression" in response.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("analysis_mode", "forecast"),
        ("correlation_method", "distance"),
        ("missing_policy", "discard"),
    ],
)
def test_analyze_rejects_unsupported_contract_options_at_schema_boundary(
    client: TestClient, field: str, value: str
) -> None:
    analysis: dict[str, object] = {
        "outcome": "ball_speed",
        "predictors": ["club_speed"],
    }
    analysis[field] = value

    response = client.post(
        "/tools/launch-monitor-analytics/analyze",
        json={"records": _records(), "analysis": analysis},
    )

    assert response.status_code == 422
    assert any(field in tuple(error["loc"]) for error in response.json()["detail"])
