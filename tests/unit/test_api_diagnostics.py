"""
Tests for API diagnostics module.

Tests cover:
- Diagnostic check execution
- Static file verification
- Route registration checks
- CORS configuration validation
- Performance metrics
"""

from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip module if dependencies aren't available
fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# Try to import the diagnostics module
try:
    from src.api.diagnostics import (  # noqa: E402
        APIDiagnostics,
        DiagnosticResult,
        get_diagnostic_endpoint_html,
    )
except ImportError as e:
    pytest.skip(f"Cannot import diagnostics module: {e}", allow_module_level=True)


class TestDiagnosticResult:
    """Tests for the DiagnosticResult dataclass."""

    def test_diagnostic_result_creation(self) -> None:
        """Test creating a DiagnosticResult."""
        result = DiagnosticResult(
            name="test_check",
            status="pass",
            message="Test passed",
            details={"key": "value"},
            duration_ms=1.5,
        )
        assert result.name == "test_check"
        assert result.status == "pass"
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert result.duration_ms == 1.5

    def test_diagnostic_result_to_dict(self) -> None:
        """Test converting DiagnosticResult to dictionary."""
        result = DiagnosticResult(
            name="test_check",
            status="warning",
            message="Warning message",
            details={"warning_code": 123},
            duration_ms=2.567,
        )
        d = result.to_dict()
        assert d["name"] == "test_check"
        assert d["status"] == "warning"
        assert d["message"] == "Warning message"
        assert d["details"]["warning_code"] == 123
        assert d["duration_ms"] == 2.57  # Rounded to 2 decimal places

    def test_diagnostic_result_default_details(self) -> None:
        """Test DiagnosticResult default values."""
        result = DiagnosticResult(
            name="simple_check",
            status="pass",
            message="OK",
        )
        assert result.details == {}
        assert result.duration_ms == 0.0


class TestAPIDiagnostics:
    """Tests for the APIDiagnostics class."""

    def test_diagnostics_initialization(self) -> None:
        """Test APIDiagnostics initialization."""
        app = FastAPI()
        diag = APIDiagnostics(app)
        assert diag.app == app
        assert diag.results == []

    def test_diagnostics_without_app(self) -> None:
        """Test APIDiagnostics without FastAPI app reference."""
        diag = APIDiagnostics()
        assert diag.app is None
        results = diag.run_all_checks()
        assert "summary" in results
        assert "checks" in results
        assert "recommendations" in results

    def test_run_all_checks_returns_summary(self) -> None:
        """Test that run_all_checks returns proper summary."""
        diag = APIDiagnostics()
        results = diag.run_all_checks()

        summary = results["summary"]
        assert "total_checks" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "warnings" in summary
        assert "status" in summary
        assert "timestamp" in summary

        # Verify counts add up
        assert (
            summary["passed"] + summary["failed"] + summary["warnings"]
            == summary["total_checks"]
        )

    def test_check_python_environment(self) -> None:
        """Test Python environment check."""
        diag = APIDiagnostics()
        result = diag.check_python_environment()

        assert result.name == "python_environment"
        assert result.status == "pass"
        assert "python_version" in result.details
        assert "platform" in result.details
        assert "executable" in result.details

    def test_check_dependencies(self) -> None:
        """Test dependencies check."""
        diag = APIDiagnostics()
        result = diag.check_dependencies()

        assert result.name == "dependencies"
        assert "dependencies" in result.details
        assert "versions" in result.details

        # FastAPI should be available since we imported it
        deps = result.details["dependencies"]
        assert deps.get("fastapi") is True

    def test_check_static_files_ui_not_built(self) -> None:
        """Test static files check when UI is not built."""
        diag = APIDiagnostics()

        # Mock the ui_path to not exist
        with patch.object(Path, "exists", return_value=False):
            result = diag.check_static_files()
            # Result will depend on actual file system state
            assert result.name == "static_files"
            assert result.status in ("pass", "fail")

    def test_check_api_routes_with_app(self) -> None:
        """Test API routes check with a FastAPI app."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "healthy"}

        diag = APIDiagnostics(app)
        result = diag.check_api_routes()

        assert result.name == "api_routes"
        assert "routes" in result.details
        assert result.details["total_routes"] >= 2

    def test_check_cors_config(self) -> None:
        """Test CORS configuration check."""
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],
            allow_methods=["*"],
        )

        diag = APIDiagnostics(app)
        result = diag.check_cors_config()

        assert result.name == "cors_config"
        assert "expected_origins" in result.details

    def test_check_engine_manager(self) -> None:
        """Test engine manager check."""
        diag = APIDiagnostics()
        result = diag.check_engine_manager()

        assert result.name == "engine_manager"
        # Status depends on whether engine manager is available
        assert result.status in ("pass", "warning", "fail")

    def test_recommendations_generation(self) -> None:
        """Test that recommendations are generated based on results."""
        diag = APIDiagnostics()
        results = diag.run_all_checks()

        assert "recommendations" in results
        assert isinstance(results["recommendations"], list)
        assert len(results["recommendations"]) > 0

    def test_diagnostic_result_timing(self) -> None:
        """Test that diagnostic checks measure duration."""
        diag = APIDiagnostics()
        result = diag.check_python_environment()

        # Duration should be non-negative
        assert result.duration_ms >= 0


class TestDiagnosticEndpointHtml:
    """Tests for HTML diagnostic report generation."""

    def test_generate_html_report(self) -> None:
        """Test HTML report generation."""
        results = {
            "summary": {
                "total_checks": 5,
                "passed": 4,
                "failed": 0,
                "warnings": 1,
                "status": "healthy",
                "timestamp": "2026-01-30T12:00:00Z",
            },
            "checks": [
                {
                    "name": "test_check",
                    "status": "pass",
                    "message": "Test passed",
                    "details": {},
                    "duration_ms": 1.0,
                }
            ],
            "recommendations": ["All systems operational"],
        }

        html = get_diagnostic_endpoint_html(results)

        assert "<!DOCTYPE html>" in html
        assert "Golf Modeling Suite" in html
        assert "healthy" in html.upper() or "HEALTHY" in html
        assert "test_check" in html
        assert "Test passed" in html

    def test_generate_html_report_with_failures(self) -> None:
        """Test HTML report with failed checks."""
        results = {
            "summary": {
                "total_checks": 3,
                "passed": 1,
                "failed": 2,
                "warnings": 0,
                "status": "degraded",
                "timestamp": "2026-01-30T12:00:00Z",
            },
            "checks": [
                {
                    "name": "failing_check",
                    "status": "fail",
                    "message": "Critical failure",
                    "details": {"error": "Something broke"},
                    "duration_ms": 5.0,
                }
            ],
            "recommendations": ["Fix the failing check"],
        }

        html = get_diagnostic_endpoint_html(results)

        assert "Critical failure" in html
        assert "degraded" in html.lower()


class TestAPIDiagnosticsPerformance:
    """Performance tests for API diagnostics."""

    def test_diagnostics_complete_in_reasonable_time(self) -> None:
        """Test that all diagnostics complete within reasonable time."""
        diag = APIDiagnostics()

        start = time.time()
        results = diag.run_all_checks()
        elapsed = time.time() - start

        # All checks should complete within 10 seconds
        assert elapsed < 10.0, f"Diagnostics took too long: {elapsed:.2f}s"

        # Verify timing data is captured
        for check in results["checks"]:
            assert "duration_ms" in check
            assert check["duration_ms"] >= 0

    def test_individual_check_performance(self) -> None:
        """Test individual check performance."""
        diag = APIDiagnostics()

        checks = [
            diag.check_python_environment,
            diag.check_dependencies,
        ]

        for check_func in checks:
            start = time.time()
            result = check_func()
            elapsed = time.time() - start

            # Each check should complete within 2 seconds
            assert elapsed < 2.0, f"{result.name} took too long: {elapsed:.2f}s"


# Module-level fixture for TestLocalServerDiagnostics
@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client for local server."""
    try:
        from src.api.local_server import create_local_app

        app = create_local_app()
        with TestClient(app, base_url="http://localhost") as test_client:
            yield test_client
    except ImportError:
        pytest.skip("local_server not available")


# Integration test with local_server
@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client for local server."""
    try:
        from src.api.local_server import create_local_app

        app = create_local_app()
        with TestClient(app, base_url="http://localhost") as test_client:
            yield test_client
    except ImportError:
        pytest.skip("local_server not available")


class TestLocalServerDiagnostics:
    """Integration tests for local_server diagnostic endpoints."""

    def test_diagnostics_endpoint_json(self, client: TestClient) -> None:
        """Test /api/diagnostics returns JSON."""
        response = client.get("/api/diagnostics")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "checks" in data
        assert "recommendations" in data

    def test_diagnostics_endpoint_html(self, client: TestClient) -> None:
        """Test /api/diagnostics/html returns HTML."""
        response = client.get("/api/diagnostics/html")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "<!DOCTYPE html>" in response.text

    def test_debug_routes_endpoint(self, client: TestClient) -> None:
        """Test /api/debug/routes returns route list."""
        response = client.get("/api/debug/routes")
        assert response.status_code == 200
        data = response.json()
        assert "routes" in data
        assert "total_routes" in data
        assert isinstance(data["routes"], list)

    def test_debug_static_endpoint(self, client: TestClient) -> None:
        """Test /api/debug/static returns static file info."""
        response = client.get("/api/debug/static")
        assert response.status_code == 200
        data = response.json()
        assert "ui_path" in data
        assert "ui_exists" in data

    def test_health_endpoint_includes_ui_status(self, client: TestClient) -> None:
        """Test /api/health includes UI availability status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ui_available" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
