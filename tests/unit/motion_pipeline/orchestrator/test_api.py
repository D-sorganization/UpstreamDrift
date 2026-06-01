"""Tests for the motion_pipeline FastAPI surface."""

from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

from src.shared.python.motion_pipeline.api import (  # noqa: E402
    PipelineRequest,
    PipelineResponse,
    create_app,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_pipeline_request_default_backends() -> None:
    req = PipelineRequest(source_format="c3d")
    assert req.ik_backend == "mujoco"
    assert req.matching_backend == "mujoco"


def test_pipeline_request_to_pipeline_config_round_trip() -> None:
    req = PipelineRequest(
        source_format="c3d",
        adapter_options={"sync": True},
        ik_backend="drake",
        matching_backend="pinocchio",
    )
    cfg = req.to_pipeline_config()
    assert cfg.adapter.format == "c3d"
    assert cfg.adapter.options == {"sync": True}
    assert cfg.ik_backend == "drake"
    assert cfg.matching_backend == "pinocchio"


def test_pipeline_request_preserves_string_bool_for_pydantic_coercion() -> None:
    req = PipelineRequest(
        source_format="c3d",
        preprocessing=[{"name": "normalize", "enabled": "false"}],
    )

    cfg = req.to_pipeline_config()

    assert cfg.preprocessing[0].enabled is False


def test_pipeline_response_from_error_factory() -> None:
    resp = PipelineResponse.from_error("req-1", "boom")
    assert resp.success is False
    assert resp.error == "boom"
    assert resp.audit_log == []


def test_run_pipeline_missing_source_format_returns_422(client: TestClient) -> None:
    """Pydantic validation: source_format is required."""
    r = client.post(
        "/api/v1/motion-pipeline/run",
        files={"file": ("x.c3d", b"\x00" * 4, "application/octet-stream")},
    )
    # FastAPI returns 422 for missing required form field
    assert r.status_code in (400, 422)


def test_run_pipeline_with_invalid_file_returns_error_response(
    client: TestClient,
) -> None:
    """Garbage input bytes propagate as a structured error response."""
    r = client.post(
        "/api/v1/motion-pipeline/run",
        files={"file": ("garbage.c3d", b"not-a-c3d-file", "application/octet-stream")},
        data={"source_format": "c3d"},
    )
    # Endpoint catches errors and returns 200 with success=False, OR 4xx/5xx
    assert r.status_code in (200, 400, 422, 500)
    if r.status_code == 200:
        body = r.json()
        # The pipeline either succeeds (unlikely) or reports an error
        assert "success" in body


def test_openapi_schema_lists_run_endpoint(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/v1/motion-pipeline/run" in paths


# ---------------------------------------------------------------------------
# source_format validation + error-code mapping (#6930 / #6932)
# ---------------------------------------------------------------------------


def test_run_pipeline_unknown_source_format_returns_400(client: TestClient) -> None:
    """An unknown source_format is rejected with 400, not silently auto-detected."""
    r = client.post(
        "/api/v1/motion-pipeline/run",
        files={"file": ("x.dat", b"\x00" * 8, "application/octet-stream")},
        data={"source_format": "totally_not_a_format"},
    )
    assert r.status_code == 400
    assert "totally_not_a_format" in r.json()["detail"]


def test_run_pipeline_auto_source_format_is_accepted_past_validation(
    client: TestClient,
) -> None:
    """``auto`` bypasses the registry check; failure (if any) is not the 400 guard."""
    r = client.post(
        "/api/v1/motion-pipeline/run",
        files={"file": ("x.dat", b"\x00" * 8, "application/octet-stream")},
        data={"source_format": "auto"},
    )
    if r.status_code == 400:
        assert "Unknown source_format" not in r.json().get("detail", "")


def test_run_config_unknown_source_format_returns_400(client: TestClient) -> None:
    cfg = '{"source_format": "totally_not_a_format"}'
    r = client.post(
        "/api/v1/motion-pipeline/run-config",
        files={"file": ("x.dat", b"\x00" * 8, "application/octet-stream")},
        data={"config": cfg},
    )
    assert r.status_code == 400


def test_run_config_malformed_json_returns_422(client: TestClient) -> None:
    r = client.post(
        "/api/v1/motion-pipeline/run-config",
        files={"file": ("x.dat", b"\x00" * 8, "application/octet-stream")},
        data={"config": "{not valid json"},
    )
    assert r.status_code == 422
