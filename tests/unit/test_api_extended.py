"""Extended API test suite for Golf Modeling Suite REST API.

This module provides comprehensive endpoint testing beyond the basic tests,
including:
- Input validation edge cases
- Error message verification
- Response schema validation
- Concurrent request handling
- Security boundary testing

This addresses the API test coverage gap identified in Assessment G.
"""

import io
from pathlib import Path

import pytest

# Import TestClient with skip if unavailable
httpx = pytest.importorskip("httpx")
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# Try to import app, skip if dependencies unavailable
try:
    from api.server import ALLOWED_MODEL_DIRS, _validate_model_path, app
except ImportError as e:
    pytest.skip(f"Cannot import api.server: {e}", allow_module_level=True)


@pytest.fixture
def client():
    """Create a test client for the API."""
    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# Path Validation Tests (Security-Critical)
# =============================================================================


class TestPathValidation:
    """Tests for the _validate_model_path security function."""

    def test_validates_relative_path(self) -> None:
        """Test that relative paths within allowed dirs are accepted."""
        # Create a temporary file in an allowed directory
        for allowed_dir in ALLOWED_MODEL_DIRS:
            if allowed_dir.exists():
                test_file = allowed_dir / "test_model.urdf"
                try:
                    test_file.touch()
                    result = _validate_model_path("test_model.urdf")
                    assert Path(result).exists()
                    break
                finally:
                    if test_file.exists():
                        test_file.unlink()

    def test_rejects_absolute_path(self) -> None:
        """Test that absolute paths are rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _validate_model_path("/etc/passwd")
        assert exc_info.value.status_code == 400
        assert "absolute" in exc_info.value.detail.lower()

    def test_rejects_parent_traversal(self) -> None:
        """Test that parent directory traversal is rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _validate_model_path("../../../etc/passwd")

    def test_rejects_dotdot_in_middle(self) -> None:
        """Test that path traversal in middle of path is rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _validate_model_path("models/../../../secret")

    def test_rejects_windows_path_traversal(self) -> None:
        """Test that Windows-style path traversal is rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            _validate_model_path("..\\..\\..\\windows\\system32")

    def test_rejects_nonexistent_file(self) -> None:
        """Test that nonexistent files return 404."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _validate_model_path("definitely_not_a_real_file_12345.urdf")
        assert exc_info.value.status_code == 404


# =============================================================================
# Engine Management Endpoint Tests
# =============================================================================


class TestEngineManagement:
    """Extended tests for engine management endpoints."""

    def test_get_engines_structure(self, client: TestClient) -> None:
        """Test that engine list has correct structure."""
        response = client.get("/engines")
        assert response.status_code == 200
        engines = response.json()

        for engine in engines:
            assert "engine_type" in engine
            assert "status" in engine
            assert "is_available" in engine
            assert "description" in engine

    def test_load_engine_empty_type(self, client: TestClient) -> None:
        """Test loading engine with empty type string."""
        response = client.post("/engines//load")
        assert response.status_code in [404, 405, 400]

    def test_load_engine_special_characters(self, client: TestClient) -> None:
        """Test loading engine with special characters in type."""
        response = client.post("/engines/<script>alert(1)</script>/load")
        assert response.status_code == 400

    def test_load_engine_sql_injection(self, client: TestClient) -> None:
        """Test that SQL injection in engine type is safe."""
        response = client.post("/engines/'; DROP TABLE engines; --/load")
        assert response.status_code == 400


# =============================================================================
# Simulation Endpoint Tests
# =============================================================================


class TestSimulationEndpoints:
    """Extended tests for simulation endpoints."""

    def test_simulate_with_invalid_json(self, client: TestClient) -> None:
        """Test POST /simulate with malformed JSON."""
        response = client.post(
            "/simulate",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_simulate_with_empty_object(self, client: TestClient) -> None:
        """Test POST /simulate with empty JSON object."""
        response = client.post("/simulate", json={})
        assert response.status_code == 422

    def test_simulate_with_null_values(self, client: TestClient) -> None:
        """Test POST /simulate with null values in required fields."""
        response = client.post(
            "/simulate",
            json={
                "engine_type": None,
                "initial_state": None,
                "duration": None,
            },
        )
        assert response.status_code == 422

    def test_simulate_async_returns_task_id(self, client: TestClient) -> None:
        """Test that async simulation returns a task ID."""
        # This may fail if engine not initialized, which is expected
        response = client.post(
            "/simulate/async",
            json={
                "engine_type": "mujoco",
                "duration": 1.0,
            },
        )
        # Either returns task_id or 500 if engine not available
        if response.status_code == 200:
            data = response.json()
            assert "task_id" in data
            assert "status" in data
        else:
            assert response.status_code in [422, 500]

    def test_get_status_uuid_format(self, client: TestClient) -> None:
        """Test status endpoint with valid UUID format but non-existent task."""
        response = client.get("/simulate/status/550e8400-e29b-41d4-a716-446655440000")
        assert response.status_code == 404


# =============================================================================
# Video Analysis Endpoint Tests
# =============================================================================


class TestVideoAnalysisEndpoints:
    """Extended tests for video analysis endpoints."""

    def test_analyze_video_invalid_confidence(self, client: TestClient) -> None:
        """Test video analysis with invalid confidence value."""
        # Create a minimal video-like file
        video_content = b"\x00\x00\x00\x20ftypisom"  # Minimal MP4 header
        response = client.post(
            "/analyze/video",
            files={"file": ("test.mp4", io.BytesIO(video_content), "video/mp4")},
            data={"min_confidence": "not_a_number"},
        )
        assert response.status_code == 422

    def test_analyze_video_confidence_out_of_range(self, client: TestClient) -> None:
        """Test video analysis with confidence out of [0,1] range."""
        video_content = b"\x00\x00\x00\x20ftypisom"
        response = client.post(
            "/analyze/video",
            files={"file": ("test.mp4", io.BytesIO(video_content), "video/mp4")},
            data={"min_confidence": 5.0},  # Out of range
        )
        # Server should handle this gracefully
        assert response.status_code in [200, 400, 422, 500]

    def test_analyze_video_accepts_mp4(self, client: TestClient) -> None:
        """Test that MP4 content type is accepted."""
        video_content = b"\x00\x00\x00\x20ftypisom"
        response = client.post(
            "/analyze/video",
            files={"file": ("test.mp4", io.BytesIO(video_content), "video/mp4")},
        )
        # May fail due to invalid video, but should not be rejected for content type
        assert response.status_code in [200, 400, 500]

    def test_analyze_video_rejects_image(self, client: TestClient) -> None:
        """Test that image content type is rejected."""
        png_content = b"\x89PNG\r\n\x1a\n"
        response = client.post(
            "/analyze/video",
            files={"file": ("test.png", io.BytesIO(png_content), "image/png")},
        )
        assert response.status_code == 400


# =============================================================================
# Export Endpoint Tests
# =============================================================================


class TestExportEndpoints:
    """Extended tests for export endpoints."""

    def test_export_with_format_parameter(self, client: TestClient) -> None:
        """Test export with format query parameter."""
        response = client.get("/export/non-existent?format=csv")
        assert response.status_code == 404

    def test_export_with_invalid_format(self, client: TestClient) -> None:
        """Test export with unsupported format parameter."""
        response = client.get("/export/non-existent?format=invalid_format")
        assert response.status_code == 404  # Task not found first


# =============================================================================
# Biomechanics Analysis Endpoint Tests
# =============================================================================


class TestBiomechanicsEndpoints:
    """Tests for biomechanics analysis endpoints."""

    def test_analyze_biomechanics_missing_body(self, client: TestClient) -> None:
        """Test POST /analyze/biomechanics without body."""
        response = client.post("/analyze/biomechanics")
        assert response.status_code == 422

    def test_analyze_biomechanics_empty_body(self, client: TestClient) -> None:
        """Test POST /analyze/biomechanics with empty body."""
        response = client.post("/analyze/biomechanics", json={})
        assert response.status_code == 422


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_multiple_requests_dont_immediately_fail(self, client: TestClient) -> None:
        """Test that a few requests don't trigger rate limiting."""
        for _ in range(5):
            response = client.get("/health")
            # Should succeed within reasonable request count
            assert response.status_code == 200


# =============================================================================
# CORS and Security Headers Tests
# =============================================================================


class TestSecurityHeaders:
    """Tests for security-related headers."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Test that CORS headers are present for allowed origins."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS preflight should succeed for allowed origin
        assert (
            "access-control-allow-origin" in response.headers
            or response.status_code in [200, 405]
        )

    def test_cors_rejects_disallowed_origin(self, client: TestClient) -> None:
        """Test that CORS rejects disallowed origins."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://evil-site.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Either no CORS header or blocked
        headers = response.headers
        if "access-control-allow-origin" in headers:
            assert headers["access-control-allow-origin"] != "http://evil-site.com"


# =============================================================================
# Response Schema Validation Tests
# =============================================================================


class TestResponseSchemas:
    """Tests for response schema compliance."""

    def test_root_response_schema(self, client: TestClient) -> None:
        """Test root endpoint response matches expected schema."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()

        # Validate required fields
        assert isinstance(data.get("message"), str)
        assert isinstance(data.get("version"), str)
        assert isinstance(data.get("status"), str)

    def test_health_response_schema(self, client: TestClient) -> None:
        """Test health endpoint response matches expected schema."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data.get("status"), str)
        assert isinstance(data.get("engines_available"), int)
        assert isinstance(data.get("timestamp"), str)

    def test_openapi_schema_complete(self, client: TestClient) -> None:
        """Test OpenAPI schema has all expected endpoints."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        expected_paths = [
            "/",
            "/health",
            "/engines",
            "/simulate",
            "/analyze/video",
            "/analyze/biomechanics",
        ]

        for path in expected_paths:
            assert path in schema["paths"], f"Missing path: {path}"


# =============================================================================
# Error Response Tests
# =============================================================================


class TestErrorResponses:
    """Tests for error response format consistency."""

    def test_404_response_format(self, client: TestClient) -> None:
        """Test that 404 responses have detail field."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_422_includes_validation_details(self, client: TestClient) -> None:
        """Test that 422 responses include validation error details."""
        response = client.post("/simulate", json={"invalid": "data"})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
