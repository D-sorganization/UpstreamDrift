"""
API Parity Tests for Golf Modeling Suite.

Comprehensive tests ensuring all API endpoints work correctly,
following TDD best practices established in test_engine_loading.py.

Tests cover:
- Core API endpoints (health, root)
- Engine management endpoints
- Simulation lifecycle endpoints
- Authentication endpoints
- Export endpoints
- Security headers
- Performance benchmarks

Fixes #1133
"""

import time

import pytest

try:
    from fastapi.testclient import TestClient
    from src.api.server import app
except ImportError as _exc:
    pytest.skip(f"API server deps not available: {_exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    """Create test client with proper application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


# ──────────────────────────────────────────────────────────────
#  Core endpoints
# ──────────────────────────────────────────────────────────────
class TestCoreEndpoints:
    """Test core API endpoints (/, /health)."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """GET / returns API information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "Golf Modeling Suite" in data["message"]
        assert "version" in data
        assert data["status"] == "running"

    def test_health_check(self, client: TestClient) -> None:
        """GET /health returns healthy status with engine count."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert isinstance(data["engines_available"], int)
        assert "timestamp" in data


# ──────────────────────────────────────────────────────────────
#  Engine endpoints
# ──────────────────────────────────────────────────────────────
class TestEngineEndpoints:
    """Test engine management API endpoints."""

    def test_list_engines(self, client: TestClient) -> None:
        """GET /engines returns all available engines."""
        response = client.get("/engines")
        assert response.status_code == 200

        data = response.json()
        assert "engines" in data
        assert "mode" in data
        assert isinstance(data["engines"], list)
        assert len(data["engines"]) > 0

        for engine in data["engines"]:
            assert "name" in engine
            assert "available" in engine
            assert "capabilities" in engine
            assert isinstance(engine["capabilities"], list)

    @pytest.mark.parametrize(
        "engine_name",
        ["mujoco", "drake", "pinocchio", "putting_green"],
    )
    def test_probe_known_engine(self, client: TestClient, engine_name: str) -> None:
        """GET /api/engines/{name}/probe returns probe data for known engines."""
        response = client.get(f"/api/engines/{engine_name}/probe")
        assert response.status_code == 200
        data = response.json()
        assert "available" in data

    def test_probe_unknown_engine(self, client: TestClient) -> None:
        """GET /api/engines/{name}/probe handles unknown engines gracefully."""
        response = client.get("/api/engines/totally_fake_engine/probe")
        assert response.status_code in [200, 400, 404]

    def test_load_engine_via_api_path(self, client: TestClient) -> None:
        """POST /api/engines/{name}/load accepts valid engine name."""
        response = client.post("/api/engines/putting_green/load")
        # May succeed (200) or fail due to missing module (400/500)
        assert response.status_code in [200, 400, 500]

    def test_load_engine_via_typed_path(self, client: TestClient) -> None:
        """POST /engines/{type}/load accepts valid engine type."""
        response = client.post("/engines/putting_green/load")
        assert response.status_code in [200, 400, 500]


# ──────────────────────────────────────────────────────────────
#  Simulation endpoints
# ──────────────────────────────────────────────────────────────
class TestSimulationEndpoints:
    """Test simulation lifecycle API endpoints."""

    def test_simulate_missing_body(self, client: TestClient) -> None:
        """POST /simulate rejects empty body."""
        response = client.post("/simulate", json={})
        assert response.status_code == 422  # Pydantic validation

    def test_simulate_invalid_engine(self, client: TestClient) -> None:
        """POST /simulate with unknown engine returns an error indicator."""
        response = client.post(
            "/simulate",
            json={"engine_type": "nonexistent", "config": {}},
        )
        # Server may accept and report error in body, or reject outright
        if response.status_code == 200:
            data = response.json()
            # If 200, the response body should signal the error
            assert "error" in data or "status" in data or data.get("success") is False
        else:
            assert response.status_code in [400, 422, 500]


# ──────────────────────────────────────────────────────────────
#  Auth endpoints
# ──────────────────────────────────────────────────────────────
class TestAuthEndpoints:
    """Test authentication API endpoints."""

    def test_register_missing_fields(self, client: TestClient) -> None:
        """POST /register rejects empty body."""
        response = client.post("/auth/register", json={})
        assert response.status_code == 422

    def test_login_missing_credentials(self, client: TestClient) -> None:
        """POST /login rejects empty body."""
        response = client.post("/auth/login", json={})
        assert response.status_code == 422

    def test_login_invalid_credentials(self, client: TestClient) -> None:
        """POST /login rejects bad credentials."""
        response = client.post(
            "/auth/login",
            json={"email": "nobody@nope.com", "password": "wrong"},
        )
        assert response.status_code in [400, 401, 422]

    def test_me_without_auth(self, client: TestClient) -> None:
        """GET /me without token returns 401/403."""
        response = client.get("/auth/me")
        assert response.status_code in [401, 403]


# ──────────────────────────────────────────────────────────────
#  Export endpoints
# ──────────────────────────────────────────────────────────────
class TestExportEndpoints:
    """Test data export API endpoints."""

    def test_export_nonexistent_task(self, client: TestClient) -> None:
        """GET /export/{task_id} returns 404 for unknown task."""
        response = client.get("/export/nonexistent-task-id")
        assert response.status_code in [404, 400]


# ──────────────────────────────────────────────────────────────
#  Security headers
# ──────────────────────────────────────────────────────────────
class TestEndpointSecurity:
    """Test API security headers and error handling."""

    def test_invalid_endpoint_returns_404(self, client: TestClient) -> None:
        """Unknown paths return 404."""
        response = client.get("/completely/made/up/path")
        assert response.status_code == 404

    def test_security_headers_present(self, client: TestClient) -> None:
        """Responses include standard security headers."""
        response = client.get("/health")
        headers = response.headers
        assert "x-content-type-options" in headers
        assert headers["x-content-type-options"] == "nosniff"
        assert "x-frame-options" in headers

    def test_content_type_json(self, client: TestClient) -> None:
        """JSON endpoints return application/json."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_request_id_header(self, client: TestClient) -> None:
        """Responses include x-request-id for tracing."""
        response = client.get("/health")
        assert "x-request-id" in response.headers


# ──────────────────────────────────────────────────────────────
#  Performance
# ──────────────────────────────────────────────────────────────
class TestAPIPerformance:
    """Basic performance smoke tests."""

    def test_health_check_fast(self, client: TestClient) -> None:
        """Health check responds in < 1 s."""
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 1.0

    def test_engine_list_reasonable_time(self, client: TestClient) -> None:
        """Engine list responds in < 2 s."""
        start = time.time()
        response = client.get("/engines")
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 2.0
