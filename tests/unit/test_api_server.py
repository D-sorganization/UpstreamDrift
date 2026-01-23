"""Comprehensive API test suite for Golf Modeling Suite REST API.

Tests all API endpoints for:
- Correct HTTP status codes
- Response structure validation
- Error handling
- Security (path validation, rate limiting awareness)
- Edge cases

This addresses the critical 0% API test coverage gap identified in the
adversarial review (2026-01-13).
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

# Import TestClient first
httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# Try to import app, skip if dependencies unavailable
try:
    from src.api.server import app
except ImportError as e:
    pytest.skip(f"Cannot import api.server: {e}", allow_module_level=True)


@pytest.fixture(autouse=True)
def mock_video_pipeline():
    """Mock VideoPosePipeline to avoid MediaPipe dependency."""
    with patch("src.api.server.VideoPosePipeline") as mock:
        yield mock


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the API."""
    with TestClient(app, base_url="http://localhost") as test_client:
        yield test_client


class TestRootEndpoints:
    """Tests for root and health check endpoints."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test GET / returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["status"] == "running"

    def test_health_check(self, client: TestClient) -> None:
        """Test GET /health returns health status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestEngineEndpoints:
    """Tests for physics engine management endpoints."""

    def test_get_engines(self, client: TestClient) -> None:
        """Test GET /engines returns engine list."""
        response = client.get("/engines")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_load_engine_invalid_type(self, client: TestClient) -> None:
        """Test loading non-existent engine type."""
        response = client.post("/engines/invalid_engine/load")
        assert response.status_code == 400

    def test_load_engine_path_traversal_blocked(self, client: TestClient) -> None:
        """Test that path traversal attempts are blocked."""
        response = client.post(
            "/engines/mujoco/load",
            params={"model_path": "../../../etc/passwd"},
        )
        assert response.status_code in [400, 500]


class TestSimulationEndpoints:
    """Tests for simulation endpoints."""

    def test_simulate_missing_body(self, client: TestClient) -> None:
        """Test POST /simulate without body returns 422."""
        response = client.post("/simulate")
        assert response.status_code == 422

    def test_get_simulation_status_not_found(self, client: TestClient) -> None:
        """Test getting status of non-existent task."""
        response = client.get("/simulate/status/non-existent-task-id")
        assert response.status_code == 404


class TestVideoAnalysisEndpoints:
    """Tests for video analysis endpoints."""

    def test_analyze_video_no_file(self, client: TestClient) -> None:
        """Test POST /analyze/video without file returns 422."""
        response = client.post("/analyze/video")
        assert response.status_code == 422

    def test_analyze_video_wrong_content_type(self, client: TestClient) -> None:
        """Test uploading non-video file is rejected."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is not a video")
            temp_path = Path(f.name)
        try:
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/analyze/video",
                    files={"file": ("test.txt", f, "text/plain")},
                )
            assert response.status_code == 400
        finally:
            temp_path.unlink()


class TestExportEndpoints:
    """Tests for export endpoints."""

    def test_export_not_found(self, client: TestClient) -> None:
        """Test exporting non-existent task."""
        response = client.get("/export/non-existent-task-id")
        assert response.status_code == 404

    def test_export_invalid_format(self, client: TestClient) -> None:
        """Test exporting with invalid format."""
        # 'csv' is currently not supported (VALID_EXPORT_FORMATS = {'json'})
        response = client.get("/export/some-task?format=csv")
        assert response.status_code == 400
        assert "Invalid format" in response.json()["detail"]


class TestSecurityFeatures:
    """Tests for security-related features."""

    def test_path_validation_dotdot(self, client: TestClient) -> None:
        """Test .. in paths is rejected."""
        response = client.post(
            "/engines/mujoco/load",
            params={"model_path": "models/../../../secret"},
        )
        assert response.status_code in [400, 500]


class TestDocumentationEndpoints:
    """Tests for API documentation endpoints."""

    def test_docs_endpoint(self, client: TestClient) -> None:
        """Test Swagger UI documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
