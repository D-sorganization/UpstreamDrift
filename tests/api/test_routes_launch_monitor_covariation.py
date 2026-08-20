"""HTTP contract tests for canonical player-covariation analysis."""

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
            "source_id": "api-source",
            "source_row": index,
            "player_id": "A" if index < 5 else "B",
            "face_angle": float(index if index < 5 else index + 5),
            "club_path": float(4 - index if index < 5 else 19 - index),
            "ball_speed": float(100 + 2 * index),
            "monitor_vendor": "TrackMan",
        }
        for index in range(10)
    ]


def _context() -> dict[str, object]:
    return {
        "player_identity": {
            "trust_level": "explicit_user_attested",
            "identifier_column": "player_id",
            "evidence": "The dataset owner attested these stable labels.",
        },
        "sources": [
            {
                "source_id": "api-source",
                "file_sha256": "2" * 64,
                "rights_status": "public_redistributable",
            }
        ],
    }


def test_covariation_contract_and_openapi_models_are_published(
    client: TestClient,
) -> None:
    response = client.get(
        "/tools/launch-monitor-analytics/contracts/player-covariation/v1"
    )
    assert response.status_code == 200
    assert response.json()["title"] == "PlayerCovariationContractV1"

    openapi = client.get("/openapi.json").json()
    operation = openapi["paths"][
        "/tools/launch-monitor-analytics/v2/player-covariation"
    ]["post"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/PlayerCovariationResultV1")


def test_covariation_api_returns_traceable_population_result(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/player-covariation",
        json={
            "records": _records(),
            "request": {
                "x_column": "face_angle",
                "y_column": "club_path",
                "player_column": "player_id",
            },
            "context": _context(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_kind"] == "selected_pair"
    assert payload["meta_analysis"]["contributor_count"] == 2
    assert len(payload["lineage"]["backing_records"]) == 10
    assert payload["lineage"]["backing_records"][0]["source_id"] == "api-source"
    assert payload["claims"]["causal_inference"] is False


def test_covariation_api_fails_closed_without_trusted_player_identity(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/player-covariation",
        json={
            "records": _records(),
            "request": {
                "x_column": "face_angle",
                "y_column": "club_path",
                "player_column": "player_id",
            },
        },
    )

    assert response.status_code == 400
    assert "trusted player identity" in response.json()["detail"]


def test_covariation_api_rejects_forbidden_pseudo_identity(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/player-covariation",
        json={
            "records": _records(),
            "request": {
                "x_column": "face_angle",
                "y_column": "club_path",
                "player_column": "session_id",
            },
            "context": {
                "player_identity": {
                    "trust_level": "explicit_user_attested",
                    "identifier_column": "session_id",
                    "evidence": "Attested, but still not person identity.",
                }
            },
        },
    )

    assert response.status_code == 422
    assert "cannot be used as player identity" in str(response.json()["detail"])


def test_pair_scan_api_returns_ranked_and_unavailable_pairs(
    client: TestClient,
) -> None:
    records = _records()
    for record in records:
        record["constant_metric"] = 1.0
    response = client.post(
        "/tools/launch-monitor-analytics/v2/player-covariation/scan",
        json={
            "records": records,
            "request": {
                "player_column": "player_id",
                "numeric_columns": [
                    "face_angle",
                    "club_path",
                    "ball_speed",
                    "constant_metric",
                ],
            },
            "context": _context(),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_kind"] == "pair_scan"
    assert payload["pair_count"] == 6
    assert payload["unavailable_pair_count"] == 3
    assert payload["ranking"][0]["rank"] == 1


def test_capabilities_advertise_population_contract(client: TestClient) -> None:
    payload = client.get("/tools/launch-monitor-analytics/capabilities").json()
    assert payload["player_covariation_contract_version"] == (
        "launch-monitor-player-covariation/1.0.0"
    )
    assert payload["population_meta_analysis"] is True
