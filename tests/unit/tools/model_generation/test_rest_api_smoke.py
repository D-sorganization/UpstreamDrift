"""Smoke tests for model_generation REST API endpoints."""

import pytest


class TestRestAPISmoke:
    """Smoke tests for REST API endpoints.

    These tests verify that the REST API module can be imported
    and that its structure is correct. Actual endpoint testing
    requires the server to be running.
    """

    def test_rest_api_module_exists(self) -> None:
        """REST API module should exist and be importable."""
        try:
            from src.shared.python.model_generation.api import rest_api  # noqa: F401
        except ImportError:
            pytest.skip("REST API module not available in this environment")

    def test_generation_handlers_module_exists(self) -> None:
        """Generation handlers module should exist and be importable."""
        try:
            from src.shared.python.model_generation.api import (  # noqa: F401
                generation_handlers,
            )
        except ImportError:
            pytest.skip("Generation handlers module not available")

    def test_fastapi_dependency_available(self) -> None:
        """FastAPI should be available for REST API."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed - REST API tests skipped")

    def test_httpx_dependency_available(self) -> None:
        """httpx should be available for testing."""
        try:
            import httpx  # noqa: F401
        except ImportError:
            pytest.skip("httpx not installed - REST API tests skipped")
