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


def _strokes_gained_payload() -> dict[str, object]:
    states = [
        {
            "lie": "fairway",
            "context": "standard",
            "target": "hole-1",
            "distance_yards": 100.0,
            "expected_strokes": 2.8,
            "standard_error": 0.1,
        },
        {
            "lie": "fairway",
            "context": "standard",
            "target": "hole-1",
            "distance_yards": 200.0,
            "expected_strokes": 3.8,
            "standard_error": 0.14,
        },
        {
            "lie": "green",
            "context": "standard",
            "target": "hole-1",
            "distance_yards": 0.0,
            "expected_strokes": 0.0,
            "standard_error": 0.0,
        },
        {
            "lie": "green",
            "context": "standard",
            "target": "hole-1",
            "distance_yards": 20.0,
            "expected_strokes": 1.5,
            "standard_error": 0.08,
        },
    ]
    from src.tools.launch_monitor_model import baseline_table_sha256

    records = [
        {
            "shot_id": f"shot-{index}",
            "start_lie": "fairway",
            "start_context": "standard",
            "target": "hole-1",
            "start_distance": 150.0 + index,
            "finish_lie": "green",
            "finish_context": "standard",
            "finish_distance": 20.0 - index,
        }
        for index in range(3)
    ]
    return {
        "records": records,
        "baseline": {
            "contract_version": "launch-monitor-strokes-gained-baseline/2.0.0",
            "baseline_id": "api-test",
            "version": "2026.1",
            "source_url": "https://example.org/method",
            "license": "test-only",
            "table_sha256": baseline_table_sha256(states),
            "states": states,
        },
        "request": {
            "start": {
                "lie_column": "start_lie",
                "context_column": "start_context",
                "target_column": "target",
                "distance_column": "start_distance",
                "distance_unit": "yd",
            },
            "finish": {
                "lie_column": "finish_lie",
                "context_column": "finish_context",
                "target_column": "target",
                "distance_column": "finish_distance",
                "distance_unit": "yd",
            },
            "shot_id_column": "shot_id",
            "min_samples": 3,
        },
    }


def test_capabilities_are_machine_readable_and_versioned(client: TestClient) -> None:
    response = client.get("/tools/launch-monitor-analytics/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["contract_version"] == "1.0.0"
    assert payload["analysis_modes"] == ["correlation", "regression", "comprehensive"]
    assert payload["aggregate_regression_allowed"] is False
    assert payload["supported_contract_versions"] == ["1.0.0", "2.0.0"]
    assert payload["source_backed_scoring"] is True
    assert payload["strokes_gained_contract_version"] == (
        "launch-monitor-strokes-gained-analysis/1.0.0"
    )
    assert payload["outcome_proxy_is_strokes_gained"] is False


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


def test_source_backed_sg_contract_and_result_are_registered(
    client: TestClient,
) -> None:
    contract = client.get("/tools/launch-monitor-analytics/contracts/strokes-gained/v1")
    assert contract.status_code == 200
    assert contract.json()["title"] == "StrokesGainedAnalysisResultV1"

    openapi = client.get("/openapi.json").json()
    operation = openapi["paths"]["/tools/launch-monitor-analytics/v2/strokes-gained"][
        "post"
    ]
    result_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    assert result_schema["$ref"].endswith("/StrokesGainedAnalysisResultV1")


def test_source_backed_sg_api_returns_governed_result(client: TestClient) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/strokes-gained",
        json=_strokes_gained_payload(),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "available"
    assert payload["metric_name"] == "source_backed_strokes_gained"
    assert payload["value_summary"]["count"] == 3
    assert payload["baseline"]["baseline_id"] == "api-test"
    assert len(payload["row_results"]) == 3
    assert payload["claims"]["is_strokes_gained"] is True


def test_outcome_proxy_api_cannot_claim_strokes_gained(client: TestClient) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/outcome-proxy",
        json={
            "records": [{"carry": 150.0, "lateral": -10.0}],
            "request": {
                "carry_column": "carry",
                "lateral_column": "lateral",
                "carry_unit": "yd",
                "lateral_unit": "yd",
                "target_distance_yards": 150.0,
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["metric_name"] == "expected_proximity_dispersion_proxy"
    assert payload["claims"]["is_strokes_gained"] is False


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


@pytest.mark.parametrize(
    "identifier_column",
    ["session_id", "club", "source_id", "filename", "row_order", "source_row"],
)
def test_v2_api_rejects_attested_player_pseudo_identity(
    client: TestClient, identifier_column: str
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/analyze",
        json={
            "records": _records(),
            "analysis": {
                "outcome": "ball_speed",
                "predictors": ["club_speed"],
                "analysis_mode": "correlation",
                "group_by": identifier_column,
            },
            "context": {
                "player_identity": {
                    "trust_level": "explicit_user_attested",
                    "identifier_column": identifier_column,
                    "evidence": "The user attested this source field.",
                }
            },
        },
    )

    assert response.status_code == 422
    assert "cannot be used as player identity" in str(response.json()["detail"])


def test_v2_api_accepts_separate_session_and_order_evidence(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/analyze",
        json={
            "records": _records(),
            "analysis": {
                "outcome": "ball_speed",
                "predictors": ["club_speed"],
                "analysis_mode": "correlation",
            },
            "context": {
                "session_identity": {
                    "trust_level": "explicit_user_attested",
                    "identifier_column": "session_id",
                    "evidence": "The owner attested the session boundaries.",
                },
                "order_evidence": {
                    "trust_level": "source_reported",
                    "order_column": "shot_id",
                    "order_kind": "source_sequence",
                    "unit": "shot",
                    "evidence": "The export preserves device shot order.",
                },
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_identity"]["identifier_column"] == "session_id"
    assert payload["order_evidence"]["order_kind"] == "source_sequence"


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
