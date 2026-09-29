"""Observability endpoint contracts for production API operation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.routes.observability import router
from src.api.server import app as server_app

REQUIRED_METRIC_NAMES = {
    "upstreamdrift_api_info",
    "upstreamdrift_api_ready",
    "upstreamdrift_api_routes_total",
    "upstreamdrift_api_engines_available",
    "upstreamdrift_api_static_files_mounted",
    "upstreamdrift_api_startup_timestamp_seconds",
}


class _EngineManagerStub:
    """Minimal engine manager contract for observability tests."""

    def get_available_engines(self) -> list[str]:
        """Return available engine identifiers."""
        return ["mujoco", "pinocchio"]


@pytest.fixture
def app() -> FastAPI:
    """Create a focused app with only observability endpoints."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Create a test client for the focused observability app."""
    with TestClient(app) as test_client:
        yield test_client


def mark_ready(app: FastAPI) -> None:
    """Populate the post-warmup state required by /readyz."""
    app.state.engine_manager = _EngineManagerStub()
    app.state.simulation_service = object()
    app.state.analysis_service = object()
    app.state.task_manager = object()
    app.state.api_started_at = 1_798_761_600.0
    app.state.static_files_mounted = True


class TestHealthz:
    """Liveness checks must stay shallow and dependency-free."""

    def test_healthz_returns_alive_before_warmup(self, client: TestClient) -> None:
        """GET /healthz succeeds before engines or services are warm."""
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "alive"}


class TestReadyz:
    """Readiness reflects whether startup warmup has completed."""

    def test_readyz_returns_503_before_warmup(self, client: TestClient) -> None:
        """GET /readyz rejects traffic before required services are present."""
        response = client.get("/readyz")

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert "missing" in body
        assert "engine_manager" in body["missing"]

    def test_readyz_returns_ready_after_warmup(
        self, app: FastAPI, client: TestClient
    ) -> None:
        """GET /readyz succeeds after startup populated required app state."""
        mark_ready(app)

        response = client.get("/readyz")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["engines_available"] == 2


class TestMetrics:
    """Prometheus exposition must include the required API metrics."""

    def test_metrics_exposes_required_names(
        self, app: FastAPI, client: TestClient
    ) -> None:
        """GET /metrics returns Prometheus text for scrape contracts."""
        mark_ready(app)

        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        body = response.text
        for metric_name in REQUIRED_METRIC_NAMES:
            assert f"# HELP {metric_name} " in body
            assert f"# TYPE {metric_name} " in body


class TestServerWiring:
    """The production API app exposes the observability router."""

    def test_server_exposes_health_readiness_and_metrics(self) -> None:
        """The main app serves health, readiness, and metrics after lifespan."""
        with TestClient(server_app) as test_client:
            health = test_client.get("/healthz")
            ready = test_client.get("/readyz")
            metrics_response = test_client.get("/metrics")

        assert health.status_code == 200
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert metrics_response.status_code == 200
        assert "upstreamdrift_api_ready 1" in metrics_response.text
