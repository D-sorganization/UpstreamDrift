"""OpenAPI and privacy-boundary tests for dataset-reference jobs."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.launch_monitor_analytics import (
    get_launch_monitor_dataset_job_service,
    router,
)
from src.api.services.launch_monitor_dataset_jobs import (
    DatasetJobCapacityError,
    DatasetJobService,
    DatasetRootRegistry,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    service = DatasetJobService(DatasetRootRegistry({}))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_launch_monitor_dataset_job_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    service.close()


def _request(root_id: str = "missing-authority") -> dict[str, object]:
    return {
        "dataset": {
            "root_id": root_id,
            "repository": "D-sorganization/Launch-Monitor-Flight-Model-Campaign",
            "commit": "a" * 40,
            "manifest_sha256": "b" * 64,
            "content_sha256": "c" * 64,
            "expected_row_count": 261_666,
        },
        "operation": {"kind": "source_summary"},
    }


def test_dataset_job_contract_and_routes_are_registered_in_openapi(
    client: TestClient,
) -> None:
    contract = client.get("/tools/launch-monitor-analytics/contracts/dataset-jobs/v1")
    assert contract.status_code == 200
    assert contract.json()["title"] == "DatasetJobRequestV1"

    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    create = paths["/tools/launch-monitor-analytics/v2/dataset-jobs"]["post"]
    assert create["responses"]["202"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("/DatasetJobStatusV1")
    assert "/tools/launch-monitor-analytics/v2/dataset-jobs/{job_id}/results" in paths


def test_dataset_job_submit_returns_structured_unavailable_without_paths(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dataset-jobs", json=_request()
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = client.get(f"/tools/launch-monitor-analytics/v2/dataset-jobs/{job_id}")
        if status.json()["status"] not in {"queued", "running"}:
            break
        time.sleep(0.01)

    payload = status.json()
    assert status.status_code == 200
    assert payload["status"] == "unavailable"
    assert payload["unavailable"]["code"] == "root_not_authorized"
    assert payload["input_row_count"] == 0
    assert "\\" not in payload["unavailable"]["message"]


def test_dataset_job_request_rejects_inline_records_and_client_paths(
    client: TestClient,
) -> None:
    request = _request()
    request["records"] = [{"ball_speed": 1.0}]
    dataset = request["dataset"]
    assert isinstance(dataset, dict)
    dataset["authority_path"] = "C:\\private\\authority"

    response = client.post(
        "/tools/launch-monitor-analytics/v2/dataset-jobs", json=request
    )

    assert response.status_code == 422
    body = response.text
    assert "records" in body
    assert "authority_path" in body


def test_capabilities_publish_dataset_job_bounds(client: TestClient) -> None:
    response = client.get("/tools/launch-monitor-analytics/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_reference_jobs"] is True
    assert payload["dataset_job_maximum_page_size"] == 200
    assert payload["dataset_job_inline_rows_allowed"] is False


def test_capacity_exhaustion_returns_structured_retryable_429(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DatasetJobService(DatasetRootRegistry({}))
    monkeypatch.setattr(
        service,
        "submit",
        lambda _request: (_ for _ in ()).throw(DatasetJobCapacityError("full")),
    )
    client.app.dependency_overrides[get_launch_monitor_dataset_job_service] = lambda: (
        service
    )
    response = client.post(
        "/tools/launch-monitor-analytics/v2/dataset-jobs", json=_request()
    )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "5"
    assert response.json()["detail"] == {
        "code": "dataset_job_capacity_exhausted",
        "message": "Dataset job capacity is temporarily exhausted.",
        "retryable": True,
    }
    service.close()


def test_router_shutdown_closes_and_clears_cached_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UPSTREAMDRIFT_LAUNCH_MONITOR_DATASET_ROOTS", raising=False)
    monkeypatch.delenv("LAUNCH_MONITOR_DATA_ROOT", raising=False)
    get_launch_monitor_dataset_job_service.cache_clear()
    service = get_launch_monitor_dataset_job_service()
    app = FastAPI()
    app.include_router(router)

    with TestClient(app):
        assert service.closed is False

    assert service.closed is True
    assert get_launch_monitor_dataset_job_service.cache_info().currsize == 0
